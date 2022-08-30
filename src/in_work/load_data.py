from .. import (read_local, read_all_raw_data_from_gcs,
               read_processed_data_from_gcs)


def load_data(local: bool = True, raw: bool = False, verbose: bool = True):
    if local:
        return read_local(verbose=verbose)
    if raw:
        return read_all_raw_data_from_gcs(verbose=verbose)
    return read_processed_data_from_gcs(verbose=verbose)
