import os
import json
import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(__file__)
CACHE_DIR = os.path.join(BASE_DIR, 'data', 'extracted_cache')


def load(name: str) -> pd.DataFrame:
    path = os.path.join(CACHE_DIR, f'{name}.parquet')
    if os.path.exists(path):
        try:
            return pd.read_parquet(path)
        except Exception as e:
            print(f'Failed to read {path}: {e}')
            return pd.DataFrame()
    return pd.DataFrame()


def pct(x: float) -> float:
    return round(100.0 * x, 1)


def main():
    eda_first_adm = load('first_admissions')
    eda_demo = load('demographics')
    eda_vitals = load('vitals_48h')
    eda_labs = load('labs_48h')
    eda_rx = load('prescriptions_48h')
    eda_proc = load('procedures_48h')
    eda_micro = load('microbiology_48h') if eda_proc.empty else pd.DataFrame()

    print('Cached tables:')
    for n, df in {
        'first_admissions': eda_first_adm,
        'demographics': eda_demo,
        'vitals_48h': eda_vitals,
        'labs_48h': eda_labs,
        'prescriptions_48h': eda_rx,
        'procedures_48h': eda_proc,
        'microbiology_48h (fallback)': eda_micro,
    }.items():
        print(f'- {n}: {len(df)} rows')

    insights = []

    # Cohort size
    n_subjects = int(eda_first_adm['subject_id'].nunique()) if not eda_first_adm.empty else 0
    insights.append(f'Cohort subjects (initial): {n_subjects}')

    # Age and gender
    if not eda_first_adm.empty and not eda_demo.empty:
        tmp = eda_first_adm[['subject_id', 'admittime']].merge(
            eda_demo[['subject_id', 'dob', 'gender']], on='subject_id', how='left'
        )
        tmp['age_years'] = (pd.to_datetime(tmp['admittime']) - pd.to_datetime(tmp['dob'])).dt.days / 365.25
        tmp['age_years'] = tmp['age_years'].clip(lower=0)
        tmp.loc[tmp['age_years'] >= 89, 'age_years'] = 90
        if not tmp['age_years'].dropna().empty:
            insights.append(
                'Age at admission (years): median {:.1f} [IQR {:.1f}-{:.1f}], >=65: {:.1f}%'.format(
                    tmp['age_years'].median(),
                    tmp['age_years'].quantile(0.25),
                    tmp['age_years'].quantile(0.75),
                    100.0 * (tmp['age_years'] >= 65).mean(),
                )
            )
        if 'gender' in tmp and not tmp['gender'].dropna().empty:
            g = tmp['gender'].value_counts(normalize=True).mul(100).round(1).to_dict()
            insights.append('Gender distribution (%): ' + ', '.join(f'{k}: {v}%' for k, v in g.items()))

    # Admission types and LOS
    if not eda_first_adm.empty:
        if 'admission_type' in eda_first_adm.columns:
            at = eda_first_adm['admission_type'].value_counts(normalize=True).mul(100).round(1).to_dict()
            insights.append('Admission types (%): ' + ', '.join(f'{k}: {v}%' for k, v in at.items()))
        los_hours = (pd.to_datetime(eda_first_adm['dischtime']) - pd.to_datetime(eda_first_adm['admittime']))\
            .dt.total_seconds() / 3600.0
        los_hours = los_hours.clip(lower=0)
        if not los_hours.dropna().empty:
            insights.append(
                'Hospital LOS (hours): median {:.1f} [IQR {:.1f}-{:.1f}], >=54h coverage: {:.1f}%'.format(
                    los_hours.median(),
                    los_hours.quantile(0.25),
                    los_hours.quantile(0.75),
                    100.0 * (los_hours >= 54).mean(),
                )
            )

    # Coverage per modality
    if n_subjects:
        def cov_head(df, name):
            if df.empty:
                return f'{name}: 0% subjects with data'
            return f'{name}: {pct(df["subject_id"].nunique() / n_subjects)}% subjects with data'
        insights.append(cov_head(eda_vitals, 'Vitals (<=48h)'))
        insights.append(cov_head(eda_labs, 'Labs (<=48h)'))
        insights.append(cov_head(eda_rx, 'Prescriptions (<=48h)'))
        if not eda_proc.empty:
            insights.append(cov_head(eda_proc, 'Procedures (<=48h)'))
        elif not eda_micro.empty:
            insights.append(cov_head(eda_micro, 'Microbiology (<=48h)'))

    # Top prescriptions
    top_drugs = None
    if not eda_rx.empty and 'drug' in eda_rx.columns:
        top_drugs = (
            eda_rx.assign(drug_low=eda_rx['drug'].astype(str).str.lower())
            .groupby('drug_low')
            .size()
            .sort_values(ascending=False)
            .head(10)
        )

    # Top procedure categories
    top_proc = None
    if not eda_proc.empty and 'ordercategoryname' in eda_proc.columns:
        top_proc = eda_proc['ordercategoryname'].value_counts().head(10)

    # Heuristic rates for key procedures
    heuristics = {}
    if not eda_proc.empty and n_subjects:
        label_col = 'item_label' if 'item_label' in eda_proc.columns else 'ordercategorydescription'
        lbl = eda_proc[label_col].astype(str).str.lower()
        def frac(patterns):
            m = lbl.str.contains('|'.join(patterns), na=False)
            return pct(eda_proc.loc[m, 'subject_id'].nunique() / n_subjects)
        heuristics['ventilation_related_%'] = frac(['vent', 'intubat', 'endotracheal', 'peep', 'tidal volume'])
        heuristics['renal_replacement_%'] = frac(['dialysis', 'crrt', 'hemodialysis'])
        heuristics['central_line_%'] = frac(['central line', 'cvc', 'subclavian', 'internal jugular', 'femoral line'])

    # Print insights
    print('\nKey EDA insights:')
    for s in insights:
        print('-', s)

    # Print top tables
    if isinstance(top_drugs, pd.Series):
        print('\nTop 10 prescribed drugs (mentions):')
        print(top_drugs.to_string())
    if isinstance(top_proc, pd.Series):
        print('\nTop 10 procedure categories:')
        print(top_proc.to_string())
    if heuristics:
        print('\nProcedure heuristics (% of subjects):')
        for k, v in heuristics.items():
            print(f'- {k}: {v}%')

    # Save json report
    report = {
        'insights': insights,
        'heuristics': heuristics,
        'top_drugs': top_drugs.to_dict() if isinstance(top_drugs, pd.Series) else {},
        'top_procedure_categories': top_proc.to_dict() if isinstance(top_proc, pd.Series) else {},
    }
    out_path = os.path.join(CACHE_DIR, 'eda_quick_report.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2)
    print(f"\nSaved JSON report to: {out_path}")


if __name__ == '__main__':
    main()

