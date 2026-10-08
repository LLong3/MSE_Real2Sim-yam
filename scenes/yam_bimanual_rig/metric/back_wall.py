"""Merge the back-wall verdict into metric_frame.json (run after build_metric.py; before review_images.py).

Adds fields only; existing keys keep their meaning. Values of the back-wall plane and the desk-wall gap stay
(flush); notes are replaced by the verdict. Evidence files (all in work/):
  wall_check.json          ZED NEURAL_LIGHT depth on the wall: planarity, yaw/lean, temporal SD      (wall_check.py)
  stereo_edges.json        classical stereo on the rectified ZED pair: desk plane, wall-end edge      (zed_stereo_pair.py, stereo_edges.py)
  wall_gap_reference.json  Pi3X reference: desk back boundary vs wall per 10 cm band; wall-end slices (wall_gap_reference.py)
  video_slot_check.json    source video + ZED image: dark line at the junction vs predicted slot width (video_slot_check.py)
"""
import json
from datetime import datetime, timezone

import numpy as np

import common

OUT = common.HERE / 'metric_frame.json'
J = json.load(open(OUT))
W = lambda p: json.load(open(common.WORK / p))
wc, se, gr, vs = W('wall_check.json'), W('stereo_edges.json'), W('wall_gap_reference.json'), W('video_slot_check.json')
y1 = J['desk']['box_y_m'][1]
chk = J['scene_camera']['checks']['zed_depth_vs_metric_back_wall']

tab = wc['wall_y_mm_by_column_and_height']['table']
near = [v['z0.00-0.05'][0] for v in tab.values() if v.get('z0.00-0.05')]
high = [v['z0.60-0.90'][0] for v in tab.values() if v.get('z0.60-0.90')]
central = [r['gap_wall_minus_desk_back_mm'] for r in gr['per_x_band'] if -2.95 < r['x'] < -1.75]
dips = {k: v['dip_width_over_slot_px_per_cm_median'] for k, v in vs['narrow_dip'].items() if k != 'zed_scene_camera'}
zd = vs['narrow_dip']['zed_scene_camera']
band85 = [v['band_for_85mm_px'] for v in vs['summary'].values()]
cl = se['classical_only_crease_test']
ej = se['edges']['glass_wall_junction']['neural_plane']

evidence = dict(
    zed_neural_depth_not_a_wall=dict(
        finding='the 85 mm comes from a depth surface that is not a wall: non-planar and yawed/leaning against the desk edge; noisy',
        plane_fit_rms_mm=wc['wall_plane_fit']['rms_mm'], yaw_vs_desk_back_edge_deg=wc['wall_plane_fit']['yaw_vs_crease_deg'],
        lean_deg=wc['wall_plane_fit']['lean_deg'],
        y_behind_plane_mm_near_crease_0_5cm=[min(near), max(near)], y_behind_plane_mm_at_0p6_0p9m=[min(high), max(high)],
        temporal_sd_mm=dict(wall=wc['temporal_sd_mm']['wall'], wall_p90=wc['temporal_sd_mm']['wall_p90'], desk=wc['temporal_sd_mm']['desk_top']),
        file='work/wall_check.json, work/wallcheck/zed_wall_y.jpg'),
    reference_local_gap=dict(
        finding='Pi3X desk-top points reach the wall along the whole desk; no points inside a slot',
        gap_mm_median=float(np.median(central)), gap_mm_range=[float(min(central)), float(max(central))], bands='11 x 10 cm, x -2.95..-1.75',
        views_seeing_wall='48-64 of 64', file='work/wall_gap_reference.json'),
    video_and_zed_dark_line=dict(
        finding='the only dark feature at the junction is a thin line; read as a slot it is under 1 cm in all 6 close source views',
        implied_gap_if_line_is_slot_cm=dips, slot_px_for_85mm_gap_in_those_views=[min(band85), max(band85)],
        zed_dip_px=zd['dip_width_px_median'], zed_slot_px_per_cm=zd['slot_px_per_cm'], zed_slot_px_for_85mm=zd['slot_band_px_for_85mm'],
        note='the ZED line (5 px) is blur-limited: upper bound 1.7 cm only. The broad soft shadow above the junction is lighting, not a slot.',
        file='work/video_slot_check.json, work/wallcheck/video_slot_*.jpg'),
    zed_classical_stereo=dict(
        finding='rectified-pair edge disparities (no neural depth) confirm the desk plane; the one textured wall feature is the painted '
                'wall end next to the glass, and it is not on the main wall face',
        rectification_dy_px=se['rectification_dy_px']['median'], baseline_m=se['baseline_m'],
        desk_side_edges_vs_desk_plane_mm=[se['edges']['desk_left_edge']['neural_plane']['z_mm_median'], se['edges']['desk_right_edge']['neural_plane']['z_mm_median']],
        wall_end_edge=dict(x_from_desk_left_end_m=ej['x_m_median'], y_behind_plane_mm_at_desk_height=cl['junction_behind_desk_back_edge_mm'],
                           y_behind_plane_mm_median_0_0p55m=ej['y_mm_median'], lean_deg=ej['lean_deg'], disparity_row_noise_px=se['edges']['glass_wall_junction']['disparity_fit_rms_px']),
        reference_at_wall_end='glass plane ~71 mm behind (x < x0 + 0.045), mixed x0 + 0.05..0.09, main wall face flush from x0 + 0.11 (work/wall_gap_reference.json wall_left_end_slices)',
        file='work/stereo_edges.json, work/wallcheck/zed_stereo_pair.npz'),
)
verdict = dict(
    question='ZED depth puts the back wall ~85 mm beyond the metric back-wall plane (plane through the desk back edge): gap, ZED bias or transform error?',
    answer='(b) ZED NEURAL_LIGHT depth bias on the textureless wall. The desk is flush with the wall to ~1 cm.',
    not_a_gap='an 85 mm gap would show as a 23 px dark slot in the ZED image and 44-57 px in the close source views; neither exists. '
              f'The reference desk top meets the wall (gap {np.median(central):.1f} mm median over 11 bands).',
    not_transform='the check is ZED-internal: the plane passes through the ZED-measured desk back edge (the camera pose is anchored on it), '
                  'so the reference transform cannot enter. The transformed reference wall sits 13-16 mm in front of the plane '
                  '(the y-offset compromise between desk edges and ZED arm bases), within the stated 2-3 cm local error.',
    wall_end_detail='at the painted wall end, next to the frosted glass (desk left end), the wall steps back: ZED classical stereo puts '
                    'the painted edge 41 mm behind the plane at desk height (x = x0 + 0.06); the glass plane is ~0.07 m behind (reference). '
                    'This is a reveal/recess at the wall end, not the wall face behind the desk.',
    options=dict(
        A_flush_adopted=dict(back_edge_to_wall_gap_m=0.0, back_wall_plane_y_m=y1,
                             residuals=dict(reference_local_gap_mm=float(np.median(central)), zed_neural_wall_median_mm=chk['median_mm'],
                                            dark_line='contact shadow (0.6-0.8 cm wide in the video)',
                                            note='arm-base-to-desk-back-edge distance: ZED 0.704/0.704 m vs reference 0.691/0.681 m (sy 1.00) '
                                                 'hints at sy ~1.02-1.04; absorbed by the arm y residuals (0.2 / 11.5 mm)')),
        B_small_gap=dict(back_edge_to_wall_gap_m=0.01, back_wall_plane_y_m=y1 + 0.01,
                         residuals=dict(reference_local_gap_mm=float(np.median(central)) - 10.0, zed_neural_wall_median_mm=chk['median_mm'] - 10.0,
                                        dark_line='the slot itself (0.6-0.8 cm)'),
                         note='largest gap consistent with the video; also fits the arm-based sy ~1.03 hint. Not adopted: needs Pi3X to have filled a 1 cm slot.')),
    settle_by='tape: slide a thin ruler or card down between the desk top back edge and the wall at 2-3 places along the desk '
              '(e.g. behind each arm); if it does not go in, the desk is flush (option A). A tape of the wall-end reveal depth next to the glass is optional.',
    evidence=evidence,
)

J['back_wall_check'] = verdict
J['desk']['back_edge_to_wall_gap_bounds_m'] = [0.0, 0.01]
J['desk']['back_edge_to_wall_gap_note'] = 'flush (option A); up to 0.01 m not excluded; see back_wall_check'
bw = J['planes']['back_wall']
bw['note'] = ('vertical, through the desk back edge; desk flush with the wall to ~1 cm (back_wall_check). The ZED depth offset '
              '(~85 mm) is NEURAL_LIGHT bias on the textureless wall. Build the wall vertical here; the Pi3X wall leans ~1.8-2.2 deg.')
bw['wall_end_x_m'] = float(J['desk']['box_x_m'][0] + 0.045)
bw['wall_end_note'] = ('painted wall ends ~0.045-0.06 m right of the desk left end; the painted end steps back (edge 0.041 m behind '
                       'the plane at desk height, ZED stereo); the frosted glass to its left is ~0.07 m behind the plane (reference)')
chk['note'] = ('ZED NEURAL_LIGHT depth bias on the textureless wall, not a gap and not a transform error (back_wall_check): '
               'the wall depth surface is non-planar (plane rms 40 mm), yawed 5 deg and leaning 6 deg vs the desk edge, '
               '20-39 mm at the crease and 69-184 mm at 0.6-0.9 m, temporal SD 12 mm vs 3 mm on the desk. Not used for the pose.')
chk['verdict'] = 'b_zed_depth_bias'
R = J['residuals_final']
R['back_wall']['reference_desk_back_edge_minus_plane_m'] = gr['desk_back_minus_plane_mm_median'] / 1000
R['back_wall']['reference_local_gap_wall_minus_desk_back_m'] = float(np.median(central)) / 1000
P = J['producer']
new_scripts = ['wall_check.py', 'zed_stereo_pair.py (raiden venv, SVO2, DEPTH_MODE.NONE)', 'stereo_edges.py', 'wall_gap_reference.py',
               'video_slot_check.py', 'back_wall.py', 'verify_metric.py']
P['scripts'] = [s for s in P['scripts'] if s not in new_scripts] + new_scripts
P['updated_utc'] = datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%MZ')
P['updated_by'] = 'claude-yam-metric-b (task yam-metric-20260929): back-wall verdict and README; values of the frame unchanged'
common.save_json(OUT, J)
print(json.dumps(dict(answer=verdict['answer'], reference_gap_mm=float(np.median(central)), dips=dips, zed_dip=zd, wall_end=evidence['zed_classical_stereo']['wall_end_edge']), indent=1))
