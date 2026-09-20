# DAWproject schema

`Project.xsd` and `MetaData.xsd` are vendored verbatim from
https://github.com/bitwig/dawproject (MIT License, Copyright (c) 2020 Bitwig).

`write_dawproject.py` validates every file it writes against `Project.xsd`.
This is not optional: it is what catches invalid enumeration values, which are
silently tolerated by some DAWs and which caused a real bug (a `role="group"`
channel, which is not a valid `mixerRole`, was imported by Cubase as an
ordinary audio track instead of a group buss).

Valid `mixerRole` values are: `regular`, `master`, `effect`, `submix`, `vca`.
A group/buss is **`submix`**.
