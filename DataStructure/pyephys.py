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
            chan_info = pd.DataFrame(RawsHeader.read())[
                ['ID', 'Name', 'NumRecords', 'SamplingFreq', 'SigUnits']]
        else:
            raise

        spikes_columns = ['ID', 'Label', 'NumUnits',
                          'ReferenceID', 'LowCutOff', 'HighCutOff', 'Threshold']
        if "/SpikesHeader" in file.root:
            SpikesHeader = file.get_node("/SpikesHeader")
            spikes_info = pd.DataFrame(SpikesHeader.read())

            if "Label" not in spikes_info.columns:
                spikes_info = spikes_info[[
                    col for col in spikes_columns if col != 'Label']]
                spikes_info['Label'] = b"default"

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
            spikes_info = pd.concat(
                [spikes_info, pd.DataFrame.from_records(test_append)])
            #  ----------------------------------------------------------------
            spikes_info = spikes_info[spikes_columns]
            data = pd.merge(chan_info, spikes_info, how="left", on="ID")

        else:
            data = chan_info.copy()
            data[[col for col in spikes_columns if col != 'ID']] = [
                np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]

        data['Label'].fillna(b"default", inplace=True)

        data['ReferenceID'] = data['ReferenceID'].apply(lambda x: str(
            x) if np.isnan(x) else str(int(x)))
    return data


def load_raws(filename, chan_ID):
    chan_ID = int(chan_ID)
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


def load_spikes(filename, chan_ID, label):
    chan_ID = int(chan_ID)
    basename, extname = os.path.splitext(filename)
    if extname == "h5raw":
        filename = ".".join(basename, "h5")

    with tables.open_file(filename, mode="r") as file:
        if "/Spikes" in file.root:
            Spikes = file.get_node("/Spikes")

            try:
                spike_chan = Spikes._f_get_child(
                    "spike" + str(chan_ID).zfill(3))
                unitInfo = pd.DataFrame(
                    spike_chan._f_get_child("UnitsHeader").read())
                timestamps = spike_chan._f_get_child("TimeStamps").read()
                waveforms = spike_chan._f_get_child("Waveforms").read()
            except tables.NodeError:
                print("The /Spikes node does not contain the spike" +
                      str(chan_ID).zfill(3) + " node.")
                return {"unitInfo": None,
                        "unitID": None,
                        "timestamps": None,
                        "waveforms": None}
            # get units id
            unitID = np.zeros(len(timestamps))
            not_zero_units = unitInfo[unitInfo["NumRecords"] > 0]
            for unit in not_zero_units.index:
                unit_h5_name = "/".join([not_zero_units.loc[unit, "H5Location"].decode(
                ), not_zero_units.loc[unit, "H5Name"].decode()])
                ind = file.get_node(unit_h5_name).read()
                unitID[ind] = not_zero_units.loc[unit, "ID"]

            return {"unitInfo": unitInfo,
                    "unitID": unitID,
                    "timestamps": timestamps,
                    "waveforms": waveforms}
