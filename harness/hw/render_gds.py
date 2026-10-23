"""Render a routed design as a mask-layer layout image with KLayout: DEF from OpenROAD, standard cells replaced by
their SkyWater sky130_fd_sc_hd GDS geometry, routing on the LEF metal layers. Writes the whole die and a close-up.
Usage: python render_gds.py <def> <out_prefix> [zoom_um]"""
import os, sys
import klayout.db as db
import klayout.lay as lay

dfile, outp = sys.argv[1], sys.argv[2]; zoom = float(sys.argv[3]) if len(sys.argv) > 3 else 25.0
L = os.path.expanduser("~/pdk/sky130A/libs.ref/sky130_fd_sc_hd")
ly = db.Layout()
opt = db.LoadLayoutOptions()
cfg = opt.lefdef_config
cfg.lef_files = [f"{L}/techlef/sky130_fd_sc_hd__nom.tlef", f"{L}/lef/sky130_fd_sc_hd.lef"]
gds = db.Layout(); gds.read(f"{L}/gds/sky130_fd_sc_hd.gds")
cfg.macro_layouts = [gds]
cfg.macro_resolution_mode = 2            # always use the GDS geometry of a macro
ly.read(dfile, opt)
top = ly.top_cell()
print("cells", top.name, "bbox um", top.dbbox(), "layers", ly.layers())

# sky130 GDS layer numbers (layer/datatype) and display colours
GDS = {(64, 20): ("nwell", 0x9ec5e8, 0.15), (65, 20): ("diff", 0x7fbf7f, 0.55), (66, 20): ("poly", 0xd9534f, 0.75),
       (66, 44): ("licon", 0x222222, 0.9), (67, 20): ("li1", 0xa47cc4, 0.55), (67, 44): ("mcon", 0x111111, 0.9),
       (68, 20): ("met1", 0x4a7fd6, 0.55), (68, 44): ("via", 0x222222, 0.9), (69, 20): ("met2", 0xe08a2c, 0.5),
       (69, 44): ("via2", 0x222222, 0.9), (70, 20): ("met3", 0x2fa36b, 0.45), (71, 20): ("met4", 0x8c6d46, 0.4),
       (72, 20): ("met5", 0x6b7d8f, 0.35), (93, 44): ("nsdm", 0, 0), (94, 20): ("psdm", 0, 0)}
NAMED = {"li1": (0xa47cc4, 0.55), "met1": (0x4a7fd6, 0.55), "met2": (0xe08a2c, 0.5), "met3": (0x2fa36b, 0.45),
         "met4": (0x8c6d46, 0.4), "met5": (0x6b7d8f, 0.35), "mcon": (0x111111, 0.9), "via": (0x222222, 0.9),
         "via2": (0x222222, 0.9), "via3": (0x222222, 0.9), "via4": (0x222222, 0.9)}

PAT = {"nwell": 3, "diff": 0, "poly": 0, "licon": 0, "li1": 2, "mcon": 0, "met1": 5, "via": 0, "met2": 9,
       "via2": 0, "met3": 5, "met4": 9, "met5": 5, "via3": 0, "via4": 0}


def nm_of(info):
    if (info.layer, info.datatype) in GDS:
        return GDS[(info.layer, info.datatype)][0]
    return (info.name or "").split(".")[0].split("_")[0]


view = lay.LayoutView()
view.set_config("background-color", "#ffffff")
view.set_config("grid-visible", "false")
view.set_config("text-visible", "false")
view.set_config("cell-box-visible", "false" if True else "true")
ci = view.show_layout(ly, False)
view.max_hier()
view.add_missing_layers()
it = view.begin_layers()
while not it.at_end():
    lp = it.current()
    info = ly.get_info(lp.layer_index()) if lp.layer_index() >= 0 else None
    style = None
    if info is not None:
        if (info.layer, info.datatype) in GDS:
            nm, col, a = GDS[(info.layer, info.datatype)]
            style = (col, a) if a > 0 else None
        else:
            base = (info.name or "").split(".")[0].split("_")[0]
            if base in NAMED and "PIN" not in (info.name or "").upper() and "LABEL" not in (info.name or "").upper():
                style = NAMED[base]
    if style is None:
        lp.visible = False
    else:
        col, a = style
        lp.fill_color = col; lp.frame_color = col
        lp.dither_pattern = PAT.get(nm_of(info), 0)
        lp.transparent = True; lp.visible = True
        lp.width = 1
    view.set_layer_properties(it, lp)
    it.next()

bb = top.dbbox()
cx, cy = bb.center().x, bb.center().y
view.zoom_box(db.DBox(cx - zoom / 2, cy - zoom / 2, cx + zoom / 2, cy + zoom / 2))
view.save_image(f"{outp}_zoom.png", 2000, 2000)
# die view: drop contacts, vias, local interconnect and well so the metal and device layers read at full scale
DIE_HIDE = {"licon", "mcon", "via", "via2", "via3", "via4", "li1", "nwell"}
it = view.begin_layers()
while not it.at_end():
    lp = it.current()
    if lp.layer_index() >= 0 and nm_of(ly.get_info(lp.layer_index())) in DIE_HIDE:
        lp.visible = False; view.set_layer_properties(it, lp)
    it.next()
view.zoom_box(db.DBox(bb.left - 2, bb.bottom - 2, bb.right + 2, bb.top + 2))
view.save_image(f"{outp}_die.png", 2400, 2400)
print("zoom_box_um", cx - zoom / 2, cy - zoom / 2, cx + zoom / 2, cy + zoom / 2, "die_um", bb.left, bb.bottom, bb.right, bb.top)
print("saved", outp)
