import widowx_calibrate
import sys
import os.path
import math

def do_calibration(mode=None, recal=None, get_vpl=None):
    wc = widowx_calibrate.widowx_calibrate()
    im = wc.img_empty_file
    if recal is None:
      if mode != "ALL" and im is not None and os.path.isfile(im):
        recal = False
      else:
        recal = True

    if recal:
      print("doing full calibration")
      wc.bootstrap_calibration()  # calibrates empty and other predefined positions, if any.
    if not recal and get_vpl is None:
      try:
        vpl = wc.calib_data["metadata"]["best_vpl"]
        vplln = wc.tt.get_vpl_lines(vpl)
        get_vpl = False
      except:
        get_vpl = False
      if mode == "ALL":
        get_vpl = True

    vpl = wc.calib_data["metadata"]["best_vpl"]
    if vpl is None:
      wc.tabletop_analysis(im)

    if mode is None:
      VX_SET = wc.config["VX_SET"]
      wc.get_mode_ranges()
      vxset = []
      for num, imgdata in enumerate(wc.calib_info):
        try:
          sw_angle = wc.get_calib_data(idx, "swivel")
          reach = wc.get_calib_data(idx, "gripper_dist")
        except:
          gpos = wc.get_calib_data(num, "gripper_position")
          sw_angle = math.atan2(gpos['Y'], gpos['X'])
          reach = wc.lna.get_dist(0, 0, gpos['Y'], gpos['X'])
          if abs(round(reach) - reach) < .1:
            reach = round(reach)
          else:
            print("idx bad reach:", num, reach, gpos)
            continue
        if reach not in vxset:
          vxset.append(reach)
      missing_reaches = list(set(VX_SET) - set(vxset))
      print("Missing reaches:", missing_reaches, VX_SET, vxset)
      for r in missing_reaches:
        if r >= wc.main_range[0] and r <= wc.main_range[1]:
          if mode is None:
            mode = "MAIN"
          elif mode != "MAIN":
            mode = "ALL"
            break
        elif r < wc.main_range[0]:
          if mode is None:
            mode = "CLOSE"
          elif mode != "CLOSE":
            mode = "ALL"
            break
        elif r > wc.main_range[1]:
          if mode is None:
            mode = "FAR"
          elif mode != "FAR":
            mode = "ALL"
            break

    print("Calib mode:", mode)
    # wc.check_and_fix_calib_entries()
    if mode in ["MAIN", "ALL", "RESUME"]:
      wc.calibrate_by_swivel("MAIN", mode)
      wc.check_and_fix_calib_entries()
      wc.calibrate_pixel_to_gripper()
    if mode in ["FAR", "ALL", "RESUME"]:
      wc.calibrate_by_swivel("FAR", mode)
      wc.check_and_fix_calib_entries()
      wc.calibrate_pixel_to_gripper()
    if mode in ["CLOSE", "ALL", "RESUME"]:
      wc.calibrate_by_swivel("CLOSE", mode)
      wc.check_and_fix_calib_entries()
      wc.calibrate_pixel_to_gripper()
    if mode in ["CALIB"]:
      wc.calibrate_pixel_to_gripper(True)
    print("CALIBRATION COMPLETE!!")

def main(params):
    NUM_PARAMS = 3 # Number of expected parameters
    if len(params) > NUM_PARAMS:
        print("Usage: run_calibrate [\"ALL\",\"RESUME\",\"MAIN\",\"FAR\",\"CLOSE\"]")
        sys.exit(1)
    # Pad the list of parameters with `False` to meet the expected number
    while (len(params) < NUM_PARAMS):
        params.append(None)
    do_calibration(*params)

if __name__ == "__main__":
    # only pass parameters to main()
    main(sys.argv[1:])
