[![DOI](https://zenodo.org/badge/398899394.svg)](https://doi.org/10.5281/zenodo.14933301)

# GridSeis

The frequency of the GB electricity grid, nominally 50 Hz, wobbles all the time. How it wobbles depends on
what is generating: a grid run on gas and nuclear is heavy with spinning machines and moves slowly; a grid
run on wind and solar is light and rings. So the frequency, which can be measured at any socket, carries
information about how clean the grid is. GridSeis is three things built on that idea, all from NESO's
published one-second frequency record:

1. **A primer** on how the frequency behaves as the generation mix changes.
2. **A digest** of what the frequency signal contains, including a carbon-from-frequency model.
3. **The weather house**: a plug-in gauge that listens to the mains hum on a short wire and runs a small
   neural network on an ESP32-S3, with no internet, to show whether the grid is cleaner or dirtier than
   usual for the week.

The name nods to geophysics, where the same frequency analysis is used on seismic signals.

## The weather house

[hardware/](hardware/) is a carbon gauge in the style of the Austrian weather houses, the ones where a
figure with an umbrella comes out when rain is coming. An ESP32-S3-Zero picks up the 50 Hz field around a
mains cable with a 20 cm wire, measures the grid frequency once a second, and every 5 minutes turns the
last half hour into a spectrum. A small network, trained on five years of NESO data and run on the chip
with TensorFlow Lite Micro, reads 13 bands of that spectrum, each compared with the board's own last seven
days, and a servo walks a wind turbine out of one door when the grid is cleaner than usual for the week
and a power station out of the other when it is dirtier. Nothing touches the mains and nothing goes
online.

![The weather house in its three positions](hardware/images/weather_house.png)

Tested on 2024 to September 2026, years it never saw in training, the network's reading follows the real
week with a match of 0.74 (1 would be perfect); when the power station is fully out the grid is dirtier
than the middle of its week 93% of the time, and when the turbine is fully out it is cleaner 93% of the
time. The [hardware README](hardware/README.md) has the parts, wiring, 3D-printable enclosure, firmware,
the tests behind those numbers and how to retrain the network.

![On-board network against real cleanliness](hardware/images/tinyml_check.png)

## The primer

[docs/grid_frequency_primer.pdf](docs/grid_frequency_primer.pdf): a short illustrated primer on how the
frequency behaves as the generation mix changes, 2021 to 2026. Its last figure is the idea the weather
house is built on: the ringing of the frequency at 4 to 10 second periods tracks carbon intensity.

![Ringing against carbon intensity, 2021 to 2026](plots/primer/fig7_ringing_vs_carbon.png)

Figures are in `plots/primer/` and the code in `scripts/primer/`: `build_stats.py` makes the half-hour
statistics from the one-second data, `main.py` draws the figures, `waves_plot.py` draws the one half hour
split into waves, and `make_pdf.py` assembles the PDF.

## The digest

[docs/listening_to_the_grid.pdf](docs/listening_to_the_grid.pdf), nine pages: what the grid frequency is
and why it moves, how the signal is split into waves, why a light grid rings, a single 5 MW machine that
can be heard inside that ringing, how that machine reads the grid's inertia, why the ringing is a carbon
gauge, how well a model given nothing but the frequency signal predicts carbon intensity on unseen months
(about half of the half-hour variation over 2025 to 2026), and whether the same test works on Continental
Europe, Ireland and the Nordic area. Sections 7 to 9 are the background to the weather house. The
investigation behind the digest (the 0.258 Hz hum, the search of distribution-network data for its
source, and the other-grid tests) is kept outside the repository.

## Data

All of it is public.

| Dataset | Source |
|---|---|
| GB system frequency, one-second readings, July 2021 to September 2026 | [NESO data portal](https://www.neso.energy/data-portal/system-frequency-data), fetched by `scripts/fetch_frequency.py` into a folder outside the repository (`D:/gridseis_hum/freq_parquet` in the scripts) |
| Historic generation mix and carbon intensity, half-hourly, 2009 to 2026 | [NESO data portal](https://www.neso.energy/data-portal/historic-generation-mix) (`df_fuel_ckan.csv`) |
| System inertia outturn, half-hourly, April 2022 onward | NESO data portal, "GB System Inertia" yearly files (primer figure 3) |

## Running the code

```bash
conda create -n grid_seis python=3.12
pip install -r requirements.txt
python scripts/fetch_frequency.py D:/gridseis_hum/freq_csv D:/gridseis_hum/freq_parquet 2026-9   # one month
```

The weather house firmware builds with PlatformIO (`hardware/firmware`, see the hardware README), and
the network is trained with TensorFlow in a separate environment (`hardware/tools/tinyml_train.py`).

## History

GridSeis began in 2025 as an XGBoost model predicting carbon intensity from 30-minute spectra of the
frequency. That version is kept as it was on the branch `archive/original-gridseis` (tag
`archive-original-gridseis`) and in the Zenodo record above. The digest's model is its successor, and
the weather house carries the idea onto a device.

## License

[MIT](LICENSE).
