"""
Chladni sinematik: karanlık salonda, kaide üzerinde kare metal plaka.
Kumun nodal çizgilerde toplanması GERÇEK bir FDTD dalga simülasyonundan gelir.

    blender --background --python render_chladni.py -- --preview   # 4 kare 640x360
    blender --background --python render_chladni.py               # 49 kare 960x540

FİZİK (bkz. chladni.py) — ekranda görünen HER şey simülasyonun çözümüdür
--------------------------------------------------------------------------
1) Plaka: 2B dalga denklemi, kare plaka, FDTD leapfrog:
        u_tt = c²∇²u − γ·u_t + f(x,y,t)
   Serbest kenar (∂u/∂n = 0) → kenarlar antinod, kum yalnız iç nodal
   çizgilere toplanır. Merkezde sürekli harmonik sürücü (geniş Gauss titrak).

2) Rezonans: sürücü frekansı mod frekanslarına (f_mn = c·√(m²+n²)/(2L))
   kilitlenir. Mod 0 = (2,0)+(0,2) [1.4142 Hz] → halka deseni.
                Mod 1 = (2,2)       [2.0000 Hz] → ızgara deseni.

3) Kum: her tane için
        v += −σ·a·ξ              (titreşim itkisi, ξ ~ N(0,1))
        v += −μ·∇a              (genlik eğimi boyunca yokuş aşağı akış)
        v += ε·ξ                (ortam gürültüsü)
        v *= exp(−ζ·dt)         (tane–plaka sürtünmesi)
        a < a_capture → tane YAPIŞIR (klasik Chladni yakalama ölçütü)
        p += v·dt
   a = yerel titreşim zarfı (sqrt(2⟨u²⟩)), gradyanı merkezi farkla.

4) Sahnedeki iki mod tek bir SÜREKLİ simülasyonda üretilir: plaka modu
   değiştirir (müzik değişir), kum eski nodal çizgilerden serbest kalır ve
   yeni çizgilere göçer. Kamera yukarı çıkarken desen değişir.

Blender animasyonu yalnızca bu kaydı taşır; tüm anahtarlar LINEAR'dır
(easing yok). Hareketin yumuşaklığı eğimden değil, gerçek sürtünme ve
sönüm terimlerinden gelir.
"""

import math
import os
import sys
import time

import numpy as np

REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO)

import bpy  # noqa: E402
from bpy_extras.object_utils import world_to_camera_view  # noqa: E402
from mathutils import Vector  # noqa: E402

import chladni as ch  # noqa: E402

# ---------------------------------------------------------------- sabitler
PLATE = 2.0                 # plaka kenarı (Blender birimi) = 2 * L_sim
PLATE_T = 0.09              # plaka kalınlığı
TOP_Z = 0.0                 # plaka üst yüzeyi z = 0
PED_H = 1.15                # kaide yüksekliği
FLOOR_Z = TOP_Z - PLATE_T - PED_H

FREQ0 = ch.mode_frequency(2, 0)   # 1.41421 Hz
FREQ1 = ch.mode_frequency(2, 2)   # 2.00000 Hz

CLIP = 49
FPS = 12
N_GRAINS = 2400
SEED = 20240517

CACHE = os.path.join(REPO, "cache")


# ------------------------------------------------------------------ yardım
def args():
    a = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    return {"preview": "--preview" in a}


def _lineer(ob):
    ad = ob.animation_data
    if not (ad and ad.action):
        return
    act = ad.action
    if hasattr(act, "fcurves"):
        fcs = list(act.fcurves)
    else:
        fcs = [fc for layer in act.layers for strip in layer.strips
               for bag in strip.channelbags for fc in bag.fcurves]
    for fc in fcs:
        for kp in fc.keyframe_points:
            kp.interpolation = "LINEAR"


def _mat(name, color, metallic=0.0, rough=0.5, emission=None, emit_strength=1.0):
    if name in bpy.data.materials:
        return bpy.data.materials[name]
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    nt = m.node_tree
    b = nt.nodes["Principled BSDF"]
    if emission is not None:
        out = nt.nodes["Material Output"]
        nt.nodes.remove(b)
        em = nt.nodes.new("ShaderNodeEmission")
        em.inputs["Color"].default_value = (*emission, 1.0)
        em.inputs["Strength"].default_value = emit_strength
        nt.links.new(em.outputs["Emission"], out.inputs["Surface"])
    else:
        b.inputs["Base Color"].default_value = (*color, 1.0)
        b.inputs["Metallic"].default_value = metallic
        b.inputs["Roughness"].default_value = rough
    return m


# ------------------------------------------------------------- simülasyon
def sim_cache_path():
    return os.path.join(CACHE, "chladni_scene.npz")


def get_scene(force=False):
    """Simülasyonu çalıştır (veya önbellekten oku)."""
    os.makedirs(CACHE, exist_ok=True)
    path = sim_cache_path()
    if os.path.exists(path) and not force:
        try:
            z = np.load(path)
            if int(z["n_frames"]) == CLIP:
                print("[sim] önbellekten yüklendi:", path, flush=True)
                return {"grains": z["grains"], "amp": z["amp"], "times": z["times"]}
        except Exception as e:  # noqa: BLE001
            print("[sim] önbellek okunamadı, yeniden hesaplanıyor:", e, flush=True)

    print("[sim] FDTD + kum simülasyonu çalışıyor ...", flush=True)
    t0 = time.time()
    res = ch.simulate_scene(FREQ0, FREQ1, n_frames=CLIP, n_grains=N_GRAINS,
                             seed=SEED, progress=lambda i, n: print(
                                 f"  kare {i}/{n}", flush=True))
    np.savez_compressed(path, grains=res["grains"], amp=res["amp"],
                        times=res["times"], n_frames=CLIP)
    print(f"[sim] bitti ({time.time() - t0:.1f} s) -> {path}", flush=True)
    return res


# ------------------------------------------------------------------ sahne
def build_plate(tex_image):
    """Kare plaka (siyah fırçalanmış çelik) + nodal desen dokusu."""
    h = PLATE / 2.0
    z0, z1 = TOP_Z - PLATE_T, TOP_Z
    verts = [(-h, -h, z0), (h, -h, z0), (h, h, z0), (-h, h, z0),
             (-h, -h, z1), (h, -h, z1), (h, h, z1), (-h, h, z1)]
    faces = [(0, 3, 2, 1),            # alt
             (4, 5, 6, 7),            # üst
             (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)]
    me = bpy.data.meshes.new("Plaka")
    me.from_pydata(verts, [], faces)
    me.update()
    uv = me.uv_layers.new(name="UVMap")
    for poly in me.polygons:
        top = poly.index == 1
        for li in poly.loop_indices:
            vi = me.loops[li].vertex_index
            x, y, _ = verts[vi]
            if top:
                uv.data[li].uv = ((x + h) / PLATE, (y + h) / PLATE)
            else:
                uv.data[li].uv = (0.0, 0.0)
    ob = bpy.data.objects.new("Plaka", me)
    bpy.context.scene.collection.objects.link(ob)

    m = bpy.data.materials.new("Celik")
    m.use_nodes = True
    nt = m.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
    # siyah, yarı mat fırçalanmış çelik: ayna gibi yansımaz, kumu öne çıkarır
    bsdf.inputs["Metallic"].default_value = 0.0
    bsdf.inputs["Roughness"].default_value = 0.55
    # düşük speküler: plaka ayna gibi parlamasın, kum öne çıksın
    try:
        bsdf.inputs["Specular IOR Level"].default_value = 0.09
    except KeyError:
        pass

    tex = nt.nodes.new("ShaderNodeTexImage")
    tex.image = tex_image
    tex.interpolation = "Cubic"

    # temel renk: antinod (koyu) -> nodal çizgi (biraz açık gri)
    ramp = nt.nodes.new("ShaderNodeValToRGB")
    ramp.color_ramp.elements[0].position = 0.0
    ramp.color_ramp.elements[0].color = (0.014, 0.015, 0.017, 1.0)
    ramp.color_ramp.elements[1].position = 1.0
    ramp.color_ramp.elements[1].color = (0.060, 0.066, 0.078, 1.0)
    nt.links.new(tex.outputs["Color"], ramp.inputs["Fac"])
    nt.links.new(ramp.outputs["Color"], bsdf.inputs["Base Color"])

    # fırçalanmış çelik: ince gürültü pürüzlülük
    noise = nt.nodes.new("ShaderNodeTexNoise")
    noise.inputs["Scale"].default_value = 140.0
    noise.inputs["Detail"].default_value = 3.0
    rough = nt.nodes.new("ShaderNodeMath")
    rough.operation = "MULTIPLY_ADD"
    rough.inputs[1].default_value = 0.14
    rough.inputs[2].default_value = 0.52
    nt.links.new(noise.outputs["Fac"], rough.inputs[0])
    nt.links.new(rough.outputs[0], bsdf.inputs["Roughness"])

    # nodal desenin karanlıkta da okunması için çok zayıf ışıma
    # nodal çizgiyi ince bir hat olarak ışıt: yalnız çok düşük genlikler
    eramp = nt.nodes.new("ShaderNodeValToRGB")
    eramp.color_ramp.elements[0].position = 0.78
    eramp.color_ramp.elements[0].color = (0, 0, 0, 1)
    eramp.color_ramp.elements[1].position = 0.97
    eramp.color_ramp.elements[1].color = (1, 1, 1, 1)
    nt.links.new(tex.outputs["Color"], eramp.inputs["Fac"])
    em = nt.nodes.new("ShaderNodeEmission")
    em.inputs["Strength"].default_value = 0.10
    nt.links.new(eramp.outputs["Color"], em.inputs["Color"])

    add = nt.nodes.new("ShaderNodeAddShader")
    nt.links.new(bsdf.outputs["BSDF"], add.inputs[0])
    nt.links.new(em.outputs["Emission"], add.inputs[1])
    nt.links.new(add.outputs[0], out.inputs["Surface"])
    me.materials.append(m)
    return ob


def build_sand():
    """Tüm kum taneleri TEK bir mesh olarak (küçük piramitler)."""
    rng = np.random.default_rng(SEED + 7)
    n = N_GRAINS
    r = rng.uniform(0.0055, 0.0120, n)
    ang = rng.uniform(0, 2 * math.pi, n)
    ca, sa = np.cos(ang), np.sin(ang)
    corners = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]], dtype=float)
    base = np.zeros((n, 4, 2))
    for k in range(4):
        cx, cy = corners[k]
        base[:, k, 0] = (cx * ca - cy * sa) * r
        base[:, k, 1] = (cx * sa + cy * ca) * r
    height = r * 0.45

    verts = np.zeros((n * 5, 3))
    faces = []
    for i in range(n):
        b = i * 5
        for k in range(4):
            verts[b + k, :2] = base[i, k]
            verts[b + k, 2] = 0.0015
        verts[b + 4, 2] = height[i] + 0.0015
        # 4 yan yüzey (normaller dışa baksın) + taban
        faces.append((b, b + 1, b + 4))
        faces.append((b + 1, b + 2, b + 4))
        faces.append((b + 2, b + 3, b + 4))
        faces.append((b + 3, b, b + 4))
        faces.append((b, b + 3, b + 2, b + 1))
    me = bpy.data.meshes.new("Kum")
    me.from_pydata(verts.reshape(-1, 3).tolist(), [], faces)
    me.update()
    ob = bpy.data.objects.new("Kum", me)
    bpy.context.scene.collection.objects.link(ob)
    m = bpy.data.materials.new("KumBeyaz")
    m.use_nodes = True
    b = m.node_tree.nodes["Principled BSDF"]
    b.inputs["Base Color"].default_value = (0.58, 0.59, 0.62, 1.0)
    b.inputs["Roughness"].default_value = 0.70
    b.inputs["Metallic"].default_value = 0.0
    me.materials.append(m)
    return ob, base.astype(np.float32), height.astype(np.float32)


def update_sand(ob, positions, base_off, height):
    """Kum mesh'ini verilen tane konumlarıyla güncelle (tüm taneler tek mesh)."""
    n = positions.shape[0]
    px = ((positions[:, 0] - 0.5) * PLATE).astype(np.float32)
    py = ((positions[:, 1] - 0.5) * PLATE).astype(np.float32)
    co = np.zeros((n * 5, 3), dtype=np.float32)
    for k in range(4):
        co[k::5, 0] = px + base_off[:, k, 0]
        co[k::5, 1] = py + base_off[:, k, 1]
        co[k::5, 2] = 0.002          # taban plaka yüzeyinin hemen üstünde
    co[4::5, 0] = px
    co[4::5, 1] = py
    co[4::5, 2] = height + 0.002
    ob.data.vertices.foreach_set("co", co.reshape(-1))
    ob.data.update()


def build_stage():
    sc = bpy.context.scene
    # karanlık salon
    w = bpy.data.worlds.new("Salon")
    sc.world = w
    w.use_nodes = True
    bg = w.node_tree.nodes["Background"]
    bg.inputs[0].default_value = (0.0055, 0.0065, 0.0105, 1.0)
    bg.inputs[1].default_value = 1.0

    # zemin
    bpy.ops.mesh.primitive_plane_add(size=60, location=(0, 0, FLOOR_Z))
    floor = bpy.context.active_object
    floor.name = "Zemin"
    fm = bpy.data.materials.new("ZeminMat")
    fm.use_nodes = True
    fb = fm.node_tree.nodes["Principled BSDF"]
    fb.inputs["Base Color"].default_value = (0.010, 0.011, 0.013, 1.0)
    fb.inputs["Roughness"].default_value = 0.45
    floor.data.materials.append(fm)

    # kaide (koyu taş)
    bpy.ops.mesh.primitive_cube_add(size=1, location=(0, 0, FLOOR_Z + PED_H / 2))
    ped = bpy.context.active_object
    ped.name = "Kaide"
    ped.scale = (PLATE * 1.16, PLATE * 1.16, PED_H)
    bpy.ops.object.transform_apply(scale=True)
    pm = bpy.data.materials.new("KaideMat")
    pm.use_nodes = True
    nt = pm.node_tree
    pb = nt.nodes["Principled BSDF"]
    pb.inputs["Base Color"].default_value = (0.052, 0.048, 0.043, 1.0)
    pb.inputs["Roughness"].default_value = 0.68
    noise = nt.nodes.new("ShaderNodeTexNoise")
    noise.inputs["Scale"].default_value = 26.0
    noise.inputs["Detail"].default_value = 6.0
    bump = nt.nodes.new("ShaderNodeBump")
    bump.inputs["Strength"].default_value = 0.16
    nt.links.new(noise.outputs["Fac"], bump.inputs["Height"])
    nt.links.new(bump.outputs["Normal"], pb.inputs["Normal"])
    ped.data.materials.append(pm)

    # plakanın altında ince metal çerçeve
    bpy.ops.mesh.primitive_cube_add(size=1, location=(0, 0, TOP_Z - PLATE_T - 0.04))
    rim = bpy.context.active_object
    rim.name = "Cerceve"
    rim.scale = (PLATE * 1.03, PLATE * 1.03, 0.08)
    bpy.ops.object.transform_apply(scale=True)
    rim.data.materials.append(
        _mat("CerceveMat", (0.09, 0.095, 0.105), metallic=0.9, rough=0.34))


def build_atmosphere():
    """Çok seyrek bir hacim saçılması: salonda hafif atmosfer."""
    bpy.ops.mesh.primitive_cube_add(size=1, location=(0, 0, 2.0))
    vol = bpy.context.active_object
    vol.name = "Atmosfer"
    vol.scale = (26.0, 26.0, 16.0)
    bpy.ops.object.transform_apply(scale=True)
    vol.visible_shadow = False
    m = bpy.data.materials.new("AtmosferMat")
    m.use_nodes = True
    nt = m.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    scat = nt.nodes.new("ShaderNodeVolumeScatter")
    scat.inputs["Density"].default_value = 0.0055
    scat.inputs["Anisotropy"].default_value = 0.35
    nt.links.new(scat.outputs["Volume"], out.inputs["Volume"])
    vol.data.materials.append(m)
    return vol


def build_lights():
    sc = bpy.context.scene

    def aim(ob, target):
        d = bpy.data.objects.new("Hedef_" + ob.name, None)
        sc.collection.objects.link(d)
        d.location = target
        c = ob.constraints.new("TRACK_TO")
        c.target = d
        c.track_axis = "TRACK_NEGATIVE_Z"
        c.up_axis = "UP_Y"

    # sert yandan ana spot
    ld = bpy.data.lights.new("AnaSpot", "SPOT")
    ld.energy = 260.0
    ld.spot_size = math.radians(66)
    ld.spot_blend = 0.22
    ld.shadow_soft_size = 0.03
    ld.color = (1.0, 0.94, 0.84)
    lo = bpy.data.objects.new("AnaSpot", ld)
    sc.collection.objects.link(lo)
    lo.location = (-2.45, -1.80, 3.30)
    aim(lo, (0.05, 0.10, 0.0))

    # karşı taraftan soğuk dolgu
    ld2 = bpy.data.lights.new("Dolgu", "SPOT")
    ld2.energy = 60.0
    ld2.spot_size = math.radians(75)
    ld2.spot_blend = 0.55
    ld2.shadow_soft_size = 0.4
    ld2.color = (0.60, 0.70, 0.95)
    lo2 = bpy.data.objects.new("Dolgu", ld2)
    sc.collection.objects.link(lo2)
    lo2.location = (2.9, 1.9, 1.9)
    aim(lo2, (0.1, 0.1, 0.0))

    # arkadan ince kenar ışığı
    ld3 = bpy.data.lights.new("Kenar", "AREA")
    ld3.energy = 90.0
    ld3.shape = "RECTANGLE"
    ld3.size = 1.6
    ld3.size_y = 0.25
    ld3.color = (0.72, 0.80, 1.0)
    lo3 = bpy.data.objects.new("Kenar", ld3)
    sc.collection.objects.link(lo3)
    lo3.location = (0.6, 3.1, 1.5)
    aim(lo3, (0.0, 0.4, 0.1))


def camera_path(t):
    """Gerçek kamera yolu: yukarı çıkış + hafif geri çekilme + azimut kayması."""
    elev = math.radians(32.0 + 18.0 * t)
    dist = 4.15 + 1.25 * t
    az = math.radians(-98.0 + 20.0 * t)
    return (dist * math.cos(elev) * math.cos(az),
            dist * math.cos(elev) * math.sin(az),
            dist * math.sin(elev) + 0.02)


def build_camera():
    sc = bpy.context.scene
    cam = bpy.data.cameras.new("Kamera")
    cam.lens = 50.0
    cam.sensor_width = 36.0
    co = bpy.data.objects.new("Kamera", cam)
    sc.collection.objects.link(co)
    hedef = bpy.data.objects.new("Hedef", None)
    sc.collection.objects.link(hedef)
    hedef.location = (0.0, 0.0, 0.02)
    c = co.constraints.new("TRACK_TO")
    c.target = hedef
    c.track_axis = "TRACK_NEGATIVE_Z"
    c.up_axis = "UP_Y"

    for f in (1, 9, 17, 25, 33, 41, CLIP):
        co.location = camera_path((f - 1) / (CLIP - 1))
        co.keyframe_insert("location", frame=f)
    _lineer(co)
    sc.camera = co
    return co, hedef


def kompozisyon_kontrol(cam, frames):
    """Sayısal kadraj kontrolü: plaka köşeleri kamerada 0..1 aralığında mı,
    özne ekranın ne kadarını kaplıyor?"""
    sc = bpy.context.scene
    h = PLATE / 2.0
    corners = [Vector((-h, -h, TOP_Z)), Vector((h, -h, TOP_Z)),
               Vector((h, h, TOP_Z)), Vector((-h, h, TOP_Z))]
    rows = []
    for f in frames:
        sc.frame_set(f)
        bpy.context.view_layer.update()
        us, vs = [], []
        for cpt in corners:
            v = world_to_camera_view(sc, cam, cpt)
            us.append(v.x)
            vs.append(v.y)
        w = max(us) - min(us)
        hgt = max(vs) - min(vs)
        inside = (min(us) >= -0.002 and max(us) <= 1.002
                  and min(vs) >= -0.002 and max(vs) <= 1.002)
        rows.append((f, min(us), max(us), min(vs), max(vs), w * hgt, inside))
    rows = []
    for f in frames:
        sc.frame_set(f)
        bpy.context.view_layer.update()
        us, vs = [], []
        for cpt in corners:
            v = world_to_camera_view(sc, cam, cpt)
            us.append(v.x)
            vs.append(v.y)
        w = max(us) - min(us)
        hgt = max(vs) - min(vs)
        inside = (min(us) >= -0.002 and max(us) <= 1.002
                  and min(vs) >= -0.002 and max(vs) <= 1.002)
        rows.append((f, min(us), max(us), min(vs), max(vs), w * hgt, inside))
    return rows


def render_setup(preview):
    sc = bpy.context.scene
    sc.render.engine = "CYCLES"
    try:
        prefs = bpy.context.preferences.addons["cycles"].preferences
        prefs.compute_device_type = "METAL"
        prefs.get_devices()
        for d in prefs.devices:
            d.use = True
        sc.cycles.device = "GPU"
    except Exception as e:  # noqa: BLE001
        print("[atlandı] GPU:", e)
    if preview:
        sc.render.resolution_x, sc.render.resolution_y = 640, 360
        sc.cycles.samples = 32
        sc.cycles.max_bounces = 6
    else:
        sc.render.resolution_x, sc.render.resolution_y = 960,  540
        sc.cycles.samples = 64
        sc.cycles.max_bounces = 8
    sc.cycles.use_denoising = True
    sc.cycles.use_adaptive_sampling = True
    sc.cycles.adaptive_threshold = 0.01
    sc.view_settings.view_transform = "AgX"
    try:
        sc.view_settings.look = "AgX - Medium High Contrast"
    except TypeError:
        looks = [i.identifier for i in
                 sc.view_settings.bl_rna.properties["look"].enum_items]
        for cand in looks:
            if "Medium High Contrast" in cand:
                sc.view_settings.look = cand
                break
    sc.view_settings.exposure = 0.0
    sc.render.image_settings.file_format = "PNG"
    sc.render.image_settings.color_mode = "RGB"
    sc.render.film_transparent = False
    sc.render.fps = FPS
    sc.frame_start = 1
    sc.frame_end = CLIP


# ---------------------------------------------------------------- ana
def main():
    opts = args()
    sc = bpy.context.scene
    for ob in list(bpy.data.objects):
        bpy.data.objects.remove(ob)

    scene_data = get_scene()
    grains = scene_data["grains"]
    amps = scene_data["amp"]
    n_grid = amps.shape[1]

    tex_img = bpy.data.images.new("NodalDesen", n_grid, n_grid, alpha=False)
    tex_img.colorspace_settings.name = "Non-Color"

    build_stage()
    build_plate(tex_img)
    sand, base_off, height = build_sand()
    build_lights()
    build_atmosphere()
    cam, hedef = build_camera()
    render_setup(opts["preview"])

    print("[kadraj] PLAKA köşeleri -> kamera (0..1 = kare içi):", flush=True)
    for (f, u0, u1, v0, v1, area, inside) in kompozisyon_kontrol(
            cam, [1, 13, 25, 37, CLIP]):
        print(f"   kare {f:2d}: u[{u0:.3f},{u1:.3f}] v[{v0:.3f},{v1:.3f}] "
              f"alan=%{100 * area:.1f}  kare içinde={inside}", flush=True)

    out = os.path.join(REPO, "renders", "preview" if opts["preview"] else "chladni")
    os.makedirs(out, exist_ok=True)
    frames = [1, 16, 30, CLIP] if opts["preview"] else range(1, CLIP + 1)

    rgba = np.ones((n_grid, n_grid, 4), dtype=np.float32)
    for f in frames:
        sc.frame_set(f)
        update_sand(sand, grains[f - 1], base_off, height)
        # nodal desen dokusu (simülasyon zarfından): düğümler = 1
        v = np.clip(1.0 - np.asarray(amps[f - 1], dtype=np.float32), 0.0, 1.0)
        rgba[:, :, 0] = v.T
        rgba[:, :, 1] = v.T
        rgba[:, :, 2] = v.T
        tex_img.pixels.foreach_set(rgba.reshape(-1))
        tex_img.update()

        path = os.path.join(out, f"frame_{f:04d}.png")
        if os.path.exists(path):
            print(f"[kare {f}] atlandı (dosya var)", flush=True)
            continue
        sc.render.filepath = path
        bpy.ops.render.render(write_still=True)
        print(f"[kare {f}] bitti", flush=True)
    print(f"== BİTTİ -> {out} ==")


main()
