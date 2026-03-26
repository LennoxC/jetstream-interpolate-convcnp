
"""
Load a multifile csv with dask.
Save as parquet with a provided partition structure.


import dask.dataframe as dd

def csv_to_parquet(input_csv_pattern, output_parquet_path, partition_cols, pd_skip_rows=0, **kwargs):
    
    Load a multifile CSV with Dask and save as Parquet with a provided partition structure.

    Parameters:
    - input_csv_pattern: Glob pattern for the input CSV files, e.g., 'data/*.csv'.
    - output_parquet_path: Path to save the output Parquet files, e.g., 'output/parquet/'.
    - partition_cols: List of column names to use for partitioning the Parquet files, e.g., ['year', 'month'].
    - pd_skip_rows: Number of rows to skip at the beginning of each CSV file.
    - **kwargs: Additional keyword arguments to pass to the to_parquet method.
    
    # Load the CSV files into a Dask DataFrame
    df = dd.read_csv(input_csv_pattern, skip_rows=pd_skip_rows)

    # Save the DataFrame as Parquet with the specified partitioning
    df.to_parquet(output_parquet_path, partition_on=partition_cols, **kwargs)

"""