import tables
import os
import pandas as pd
import numpy as np


def load_pyephys(filename):
    basename, extname = os.path.splitext(filename)
    if extname == "h5raw":
        filename = ".".join(basename, "h5")

    with tables.open_file(filename, mode="r") as file:
        # FileHeader = file.get_node("/FileHeader")
        if "/RawsHeader" in file.root:
            RawsHeader = file.get_node("/RawsHeader")
            ChanInfo = pd.DataFrame(RawsHeader.read())[
                ['ID', 'Name', 'NumRecords', 'SamplingFreq', 'SigUnits']]
        else:
            raise

        spikes_columns = ['ID', 'Label', 'NumUnits',
                          'ReferenceID', 'LowCutOff', 'HighCutOff', 'Threshold']
        if "/SpikesHeader" in file.root:
            SpikesHeader = file.get_node("/SpikesHeader")
            SpikesInfo = pd.DataFrame(SpikesHeader.read())

            if "Label" not in SpikesInfo.columns:
                SpikesInfo = SpikesInfo[[
                    col for col in spikes_columns if col != 'Label']]
                SpikesInfo['Label'] = b"default"

            # multi-label test ----------------------------------------------------------------
            test_append = [{
                'ID': 0,
                'Label': b"0test",
                'NumUnits': .0,
                'ReferenceID': np.NaN,
                'LowCutOff': .0,
                'HighCutOff': .0,
                'Threshold': .0,
            }]
            SpikesInfo = pd.concat(
                [SpikesInfo, pd.DataFrame.from_records(test_append)])
            #  ----------------------------------------------------------------
            SpikesInfo = SpikesInfo[spikes_columns]
            data = pd.merge(ChanInfo, SpikesInfo, how="left", on="ID")

        else:
            data = ChanInfo.copy()
            data[[col for col in spikes_columns if col != 'ID']] = [
                np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]

        data['Label'].fillna(b"default", inplace=True)

    return data


def load_raws(filename, chan_ID):
    basename, extname = os.path.splitext(filename)
    if extname == "h5raw":
        filename = ".".join(basename, "h5")

    with tables.open_file(filename, mode="r") as file:
        if "/Raws" in file.root:
            Raws = file.get_node("/Raws")
            with tables.open_file(Raws(mode='r')._v_file.filename, mode="r") as rawsfile:
                data = rawsfile.get_node("/Raws/raw" + str(chan_ID).zfill(3))
                return data[:]
        else:
            raise
