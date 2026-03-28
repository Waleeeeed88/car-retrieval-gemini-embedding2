# Image Audit

- Snapshot date: March 28, 2026
- Cars reviewed: 110
- Objective flags: 2
- Heuristic review flags: 1

## Objective Flags
- `honda_pilot_2025`: low detail / possible blur (2880x1420)
- `mazda_cx_70_2025`: user-provided replacement is cleanly framed, but the native source image is still very small and may look soft after scaling

## Heuristic Review Flags
- `ford_bronco_2026`: user-provided replacement is cleanly framed, but the underlying source file is still very small and may look soft compared with the rest of the dataset

## Manual Review Notes
- `bmw_3_series_2026`, `bmw_5_series_2026`, `bmw_7_series_2026`, `bmw_i4_2026`, `bmw_i5_2026`, `bmw_ix_2026`, `bmw_x1_2026`, `bmw_x3_2026`, `bmw_x5_2026`, and `bmw_x7_2026`: replaced from the local `img/` drop and normalized to the standard `front.jpg` canvas. The old webpage hero screenshots with UI overlays are gone.
- `ford_bronco_2026`, `ford_escape_2026`, and `ford_mustang_2026`: replaced with user-provided exterior images and normalized to the standard `front.jpg` canvas.
- `hyundai_kona_2026`, `hyundai_santa_fe_2026`, `hyundai_sonata_2026`, and `hyundai_tucson_hybrid_2026`: refreshed from cleaner official renders. The old tiny-thumbnail-on-canvas issue is resolved.
- `mazda_cx_70_2025`: replaced from the local `img/` drop and normalized to the standard `front.jpg` canvas. The old wrong interior-seat image is gone.
- `mercedes_benz_glb_2026`: refreshed from an exterior vehicle render. The old dashboard/interior image is gone.
- `nissan_kicks_2026`, `nissan_leaf_2026`, `nissan_murano_2026`, and `nissan_sentra_2026`: refreshed from cleaner official renders and no longer fail the earlier size/composition checks.
- `toyota_4runner_2026`, `toyota_camry_2026`, `toyota_corolla_2026`, `toyota_highlander_2026`, `toyota_prius_2026`, `toyota_rav4_2026`, `toyota_sequoia_2026`, `toyota_sienna_2026`, `toyota_tacoma_2026`, and `toyota_tundra_2026`: refreshed from white-background Toyota jelly renders. The previous black matte / cutout artifact issue is resolved.

## Notes
- Objective checks cover missing files, unreadable images, low resolution, low-detail outliers, and extreme aspect ratios.
- Heuristic flags are candidates for weird backgrounds, blocky studio backdrops, or generally unclear car presentation.
- The Ford set has been refreshed, but `ford_bronco_2026` may still benefit from a larger native source file if you want it to match the sharpness of the rest of the cleaned renders.
- The BMW set and `mazda_cx_70_2025` now use the local clean replacements from `img/`, but most of those source files are still small natively, so they may look softer than the higher-resolution official renders.
