# InsightFace ArcFace checkpoints

AIR in the paper uses eight open-source face recognition surrogate models:

- r18 MS1MV3 ArcFace
- r18 Glint360K CosFace
- r34 MS1MV3 ArcFace
- r34 Glint360K CosFace
- r50 MS1MV3 ArcFace
- r50 Glint360K CosFace
- r100 MS1MV3 ArcFace
- r100 Glint360K CosFace

Place each `backbone.pth` under the paths referenced in `scripts/run_air_full.sh`, or edit that script to match your local checkpoint layout.
