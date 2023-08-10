# Copyright (c) 2022 MetPy Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
from datetime import datetime
import json

from js import console, document
import matplotlib.pyplot as plt
import numpy as np
import numpy.ma as ma
import pandas as pd
from pyodide.http import open_url

import metpy.calc as mpcalc
from metpy.plots import add_metpy_logo, add_unidata_logo, SkewT
from metpy.units import units, pandas_dataframe_to_unit_arrays


def get_data_clicked():
    station = Element('station_id').value
    dt = datetime.fromisoformat(Element('datetime').value)
    console.log(f'Data for {station} at {dt} requested')
    try:
        plotter.get_data(station, dt)
    except Exception as e:
        console.log(f'Error fetching data: {str(e)}')


def show_profile_clicked():
    is_checked = document.querySelector("#showprofile").checked
    plotter.enable_profile(is_checked)


class Plotter:
    def __init__(self, div):
        self._div = div
        self._data = {}
        self._init_plot()

    def _init_plot(self):
        self._fig = plt.Figure(figsize=(5.5, 5.5), dpi=200)
        self.skew = SkewT(self._fig, rect=(0.09, 0.04, 0.88, 0.94))
        self.skew.ax.set_ylim(1050, 100)
        self.skew.ax.set_xlim(-50, 40)
        self.skew.plot_dry_adiabats(alpha=0.4)
        self.skew.plot_moist_adiabats(alpha=0.4)
        self.skew.plot_mixing_lines(alpha=0.4)
        self.skew.ax.set_xlabel('Temperature (\N{DEGREE CELSIUS})')
        self.skew.ax.set_ylabel('Pressure (mb)')
        self._temp_line, = self.skew.plot([], [], 'tab:red')
        self._dewp_line, = self.skew.plot([], [], 'tab:blue')
        self._prof_line, = self.skew.plot([], [], 'black')
        self._prof_line.set_visible(False)

        self._barb_locs = np.arange(100, 1000, 50) * units('mbar')
        u = np.full_like(self._barb_locs, np.nan)
        self._barbs = self.skew.plot_barbs(self._barb_locs, u, u)
        self._barbs.set_clip_on(False)

        # Add the MetPy logo!
        add_metpy_logo(self._fig, 110, 125, size='large')
        add_unidata_logo(self._fig, 260, 125, size='large')

        self.draw()

    def get_data(self, station, dt):
        self.station = station
        self.dt = dt
        fobj = open_url(f'https://mesonet.agron.iastate.edu/json/raob.py?ts={dt:%Y%m%d%H00}&station={station}')
        json_data = json.loads(fobj.read())
        data = {}
        for profile in json_data['profiles']:
            for pt in profile['profile']:
                for field in ('drct', 'dwpc', 'hght', 'pres', 'sknt', 'tmpc'):
                    data.setdefault(field, []).append(np.nan if pt[field] is None
                                                      else pt[field])
                for field in ('station', 'valid'):
                    data.setdefault(field, []).append(np.nan if profile[field] is None
                                                      else profile[field])

        # Make sure that the first entry has a valid temperature and dewpoint
        idx = np.argmax(~(np.isnan(data['tmpc']) | np.isnan(data['dwpc'])))

        # Stuff data into a pandas dataframe
        df = pd.DataFrame()
        df['pressure'] = ma.masked_invalid(data['pres'][idx:])
        df['height'] = ma.masked_invalid(data['hght'][idx:])
        df['temperature'] = ma.masked_invalid(data['tmpc'][idx:])
        df['dewpoint'] = ma.masked_invalid(data['dwpc'][idx:])
        df['direction'] = ma.masked_invalid(data['drct'][idx:])
        df['speed'] = ma.masked_invalid(data['sknt'][idx:])
        df['station'] = data['station'][idx:]
        df['time'] = [datetime.strptime(valid, '%Y-%m-%dT%H:%M:%SZ')
                      for valid in data['valid'][idx:]]

        df['temperature'] = df['temperature'].interpolate()
        df['dewpoint'] = df['dewpoint'].interpolate()

        df['u_wind'], df['v_wind'] = mpcalc.wind_components(units.Quantity(df['speed'].values, 'knot'),
                                                            units.Quantity(df['direction'].values, 'degree'))

        # Drop any rows with all NaN values for T, Td, winds
        df = df.dropna(subset=('temperature', 'dewpoint', 'direction', 'speed',
                               'u_wind', 'v_wind'), how='all').reset_index(drop=True)

        self._data = pandas_dataframe_to_unit_arrays(df,
            {'pressure': 'hPa',
                    'height': 'meter',
                    'temperature': 'degC',
                    'dewpoint': 'degC',
                    'direction': 'degrees',
                    'speed': 'knot',
                    'u_wind': 'knot',
                    'v_wind': 'knot',
                    'station': None,
                    'time': None})

        self.update_data()

    def update_data(self):
        self.skew.ax.set_title(f'Sounding for {self.station} at {self.dt:%H}Z on {self.dt:%Y/%m/%d}')
        pressure = self._data['pressure']
        self._temp_line.set_data(self._data['temperature'].m, pressure.m)
        self._dewp_line.set_data(self._data['dewpoint'].m, pressure.m)

        prof_press, _, _, prof_temp = mpcalc.parcel_profile_with_lcl(self._data['pressure'],
                                                                     self._data['temperature'],
                                                                     self._data['dewpoint'])
        self._prof_line.set_data(prof_temp.to('degC').m, prof_press.to('hPa').m)

        ix = mpcalc.resample_nn_1d(pressure, self._barb_locs)
        self._barbs.set_UVC(self._data['u_wind'].m[ix], self._data['v_wind'].m[ix])

        self.draw()

    def enable_profile(self, enabled):
        self._prof_line.set_visible(enabled)
        self.draw()

    def draw(self):
        Element(self._div).write(self._fig)


plotter = Plotter('skewt')