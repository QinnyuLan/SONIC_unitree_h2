# SMPL Demos Matched To The H2 Reference Set

This reference set uses the same eight motion names and timelines as
`reference/example_h2/deploy`, but selects the native SMPL encoder branch.

The source SMPL motions are copied from:

`/mnt/nvme3/qingyul/seed/smplx-sonic/data/smpl_filtered`

on `m10.precognition.team`. The deploy manifest sets `reference_type` to
`smpl`, so the full release encoder uses selector 2.
