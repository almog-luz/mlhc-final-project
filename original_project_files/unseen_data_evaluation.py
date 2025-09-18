def run_pipeline_on_unseen_data(subject_ids ,con):
  """
  Run your full pipeline, from data loading to prediction.

  :param subject_ids: A list of subject IDs of an unseen test set.
  :type subject_ids: List[int]

  :param con: A DuckDB connection object.
  :type con: duckdb.connection.Connection

  :return: DataFrame with the following columns:
              - subject_id: Subject IDs, which in some cases can be different due to your analysis.
              - mortality_proba: Prediction probabilities for mortality.
              - prolonged_LOS_proba: Prediction probabilities for prolonged length of stay.
              - readmission_proba: Prediction probabilities for readmission.
  :rtype: pandas.DataFrame
  """
  raise NotImplementedError('You need to implement this function')