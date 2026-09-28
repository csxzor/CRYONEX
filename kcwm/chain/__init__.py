"""Host-level kill-chain layer: per-host memory of attack stages over days, and a forecast
of each host's next ATT&CK stage.

The network-level world model sees the last ~10 minutes of the whole network. A real APT
moves one stage at a time over hours or days (DAPT2020: reconnaissance on Tuesday, foothold
on Wednesday, lateral movement on Thursday, exfiltration on Friday, all through one host).
This layer keeps, for every internal host, what it has been through, and forecasts where it
goes next.

* ``hosts``: per-host 10-second activity table from flows.
* ``targets``: per-host stage episodes, chain steps and forecast targets.
* ``tracker``: causal per-host (and network-wide) kill-chain memory.
* ``model``: stage nowcast per host window, next-stage and time-to-next-stage forecasters.
"""
