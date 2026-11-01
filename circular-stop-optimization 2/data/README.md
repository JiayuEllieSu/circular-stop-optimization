# Data sources

The `observed/` CSVs transcribe aggregate observations from:

Videa, A., and Wang, Y. (2021). *Inference of Transit Passenger Counts and Waiting Time Using Wi-Fi Signals*. [Full report](https://rosap.ntl.bts.gov/view/dot/68758/dot_68758_DS1.pdf). DOI: [10.15788/1715288737](https://doi.org/10.15788/1715288737).

- `orange_timing_long.csv`: published Orangeline interstop and stopped-duration records. October observations are used for fitting; January 7–11, 2019 observations for evaluation.
- `orange_survey_od.csv`: pooled survey origin–destination counts. The 19 survey stop identifiers have no verified correspondence with the 20 timing-table identifiers and are not used for stop redesign.
- `published_summary_checks.csv`: reported statistics used to check numerical transcription.
- `provenance.json`: table sources, date discrepancies, missing-data treatment, and extraction limitations.

The observed data are attributed to their original authors. The repository's MIT license applies to the project code, not to third-party source material. The observations support timing prediction comparisons for an existing service, not measured savings from relocated stops.
