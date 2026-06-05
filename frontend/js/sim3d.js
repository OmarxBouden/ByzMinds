/* sim3d.js — the agentic civilization as a night city (Three.js).
   Each agent is a tower on a public plaza; tower height = disposition strength;
   lit-window facade glows in the dial colour (glass-box) or uniform honest-blue
   (public). A private channel is a conduit arcing between its members. Speeches
   are light pulses (public on the plaza, private inside the conduit); votes are
   green/red beacons. The Public<->Glass-box toggle is the hero: in public the
   conduits vanish and every tower glows the same — colluders are indistinguishable
   from honest reviewers; glass-box reveals the wiring and the true dispositions.
   Exposes window.Sim3D.scene(data, tick, mode). window.Sim is the 2D fallback if
   this never initialises (no WebGL / CDN). */
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { EffectComposer } from "three/addons/postprocessing/EffectComposer.js";
import { RenderPass } from "three/addons/postprocessing/RenderPass.js";
import { UnrealBloomPass } from "three/addons/postprocessing/UnrealBloomPass.js";
import { CSS2DRenderer, CSS2DObject } from "three/addons/renderers/CSS2DRenderer.js";

// Curated, slightly desaturated — read as a designed palette, not stock neon.
const DIAL = {
  collude: 0xff3d6e, deceive: 0xff8a3d, authority: 0x3d8bff,
  bandwagon: 0x35d6a0, sycophancy: 0xb96cff, free_ride: 0xf2c14e,
};
const HONEST = 0x4f74b8, PRV = 0xff3d6e, ACCEPT = 0x39d98a, REJECT = 0xff5a6e;
const BG = 0x05070f;

const S = {
  ready: false, data: null, mode: "glassbox",
  towers: {}, conduits: [], pulses: [], beams: [], pos: {},
};

/* A procedural skyscraper facade: dark stone with a grid of windows, most lit.
   Used as an emissiveMap so the lit windows glow in the tower's dial colour while
   the stone stays dark — this is what makes a box read as a building. */
function facadeTexture(seed = 1) {
  const cols = 6, rows = 18, cw = 22, ch = 22, pad = 7;
  const c = document.createElement("canvas");
  c.width = cols * cw; c.height = rows * ch;
  const g = c.getContext("2d");
  g.fillStyle = "#05070e"; g.fillRect(0, 0, c.width, c.height);
  let s = seed * 9301 + 49297;
  const rnd = () => ((s = (s * 9301 + 49297) % 233280) / 233280);
  for (let y = 0; y < rows; y++) for (let x = 0; x < cols; x++) {
    const r = rnd();
    // ~62% lit; lit windows brighter (full dial colour via emissive), rest dim.
    const v = r > 0.38 ? 150 + Math.floor(rnd() * 105) : 14 + Math.floor(rnd() * 22);
    g.fillStyle = `rgb(${v},${v},${v})`;
    g.fillRect(x * cw + pad, y * ch + pad, cw - 2 * pad, ch - 2 * pad);
  }
  const t = new THREE.CanvasTexture(c);
  t.colorSpace = THREE.SRGBColorSpace;
  t.wrapS = t.wrapT = THREE.RepeatWrapping;
  return t;
}

function init() {
  const host = document.getElementById("stage3d");
  if (!host) return false;
  let renderer;
  try { renderer = new THREE.WebGLRenderer({ antialias: true }); }
  catch (e) { return false; }
  const W = host.clientWidth || 900, H = host.clientHeight || 560;

  const scene = new THREE.Scene();
  scene.background = new THREE.Color(BG);
  scene.fog = new THREE.FogExp2(BG, 0.028);

  const camera = new THREE.PerspectiveCamera(44, W / H, 0.1, 400);
  camera.position.set(1.5, 8.5, 14);

  renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
  renderer.setSize(W, H);
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 1.0;
  host.appendChild(renderer.domElement);

  const labelRenderer = new CSS2DRenderer();
  labelRenderer.setSize(W, H);
  labelRenderer.domElement.style.cssText = "position:absolute;inset:0;pointer-events:none";
  host.appendChild(labelRenderer.domElement);

  const controls = new OrbitControls(camera, labelRenderer.domElement);
  controls.target.set(0, 1.6, 0);
  controls.enableDamping = true; controls.dampingFactor = 0.08;
  controls.autoRotate = true; controls.autoRotateSpeed = 0.35;
  controls.enablePan = false; controls.minDistance = 9; controls.maxDistance = 30;
  controls.maxPolarAngle = Math.PI * 0.48;
  controls.domElement.style.pointerEvents = "auto";

  scene.add(new THREE.AmbientLight(0x29406e, 0.55));
  const key = new THREE.PointLight(0x77a6ff, 45, 60); key.position.set(7, 14, 7); scene.add(key);
  const fill = new THREE.PointLight(0xff5e8a, 16, 50); fill.position.set(-9, 5, -5); scene.add(fill);
  const moon = new THREE.DirectionalLight(0x9fb6e8, 0.25); moon.position.set(-6, 18, -12); scene.add(moon);

  // plaza: dark metal disc, faint polar grid, lit perimeter ring
  const plaza = new THREE.Mesh(
    new THREE.CylinderGeometry(7.4, 7.4, 0.3, 80),
    new THREE.MeshStandardMaterial({ color: 0x080f20, metalness: 0.7, roughness: 0.45 }));
  plaza.position.y = -0.15; scene.add(plaza);
  const grid = new THREE.PolarGridHelper(7, 8, 5, 64, 0x16243f, 0x0f1a30);
  grid.position.y = 0.02; scene.add(grid);
  const ring = new THREE.Mesh(new THREE.TorusGeometry(7.4, 0.035, 8, 96),
    new THREE.MeshBasicMaterial({ color: 0x2e6bbf }));
  ring.rotation.x = Math.PI / 2; ring.position.y = 0.04; scene.add(ring);

  // distant skyline: dark silhouettes with a few lit windows; gives the plaza a city.
  const sky = new THREE.Group();
  const facade = facadeTexture(7);
  for (let i = 0; i < 56; i++) {
    const ang = (i / 56) * Math.PI * 2 + (i % 3) * 0.11;
    const rad = 13 + (i * 37 % 9);
    const h = 4 + (i * 53 % 13);
    const w = 1.4 + (i % 4) * 0.5;
    const tx = facade.clone(); tx.repeat.set(1, Math.round(h / 2.2)); tx.needsUpdate = true;
    const m = new THREE.Mesh(new THREE.BoxGeometry(w, h, w),
      new THREE.MeshStandardMaterial({ color: 0x070c18, metalness: 0.4, roughness: 0.7,
        emissive: 0x24406e, emissiveIntensity: 0.5, emissiveMap: tx }));
    m.position.set(Math.cos(ang) * rad, h / 2, Math.sin(ang) * rad);
    sky.add(m);
  }
  scene.add(sky);

  const composer = new EffectComposer(renderer);
  composer.addPass(new RenderPass(scene, camera));
  // tamed bloom: glow the windows, don't blow out whole towers
  const bloom = new UnrealBloomPass(new THREE.Vector2(W, H), 0.62, 0.85, 0.32);
  composer.addPass(bloom);

  const dyn = new THREE.Group(); scene.add(dyn);   // per-tick event layer
  const cityGroup = new THREE.Group(); scene.add(cityGroup);

  S.three = { scene, camera, renderer, labelRenderer, controls, composer, dyn, cityGroup, host, facade };
  S.ready = true;

  window.addEventListener("resize", onResize);
  function onResize() {
    const w = host.clientWidth, h = host.clientHeight || 560;
    camera.aspect = w / h; camera.updateProjectionMatrix();
    renderer.setSize(w, h); labelRenderer.setSize(w, h); composer.setSize(w, h);
  }

  (function loop() {
    requestAnimationFrame(loop);
    const t = performance.now() * 0.001;
    for (const c of S.conduits) c.material.opacity = 0.55 + 0.35 * Math.sin(t * 2.5 + c.userData.ph);
    for (const p of S.pulses) { p.position.y += 0.055; p.material.opacity = Math.max(0, p.material.opacity - 0.013); }
    for (const b of S.beams) b.material.opacity = 0.45 + 0.35 * Math.sin(t * 4 + b.userData.ph);
    controls.update();
    composer.render();
    labelRenderer.render(scene, camera);
  })();
  return true;
}

function buildCity(data) {
  const { cityGroup, facade } = S.three;
  cityGroup.clear();
  S.towers = {}; S.conduits = []; S.pos = {};
  const n = data.agents.length, R = 4.6;
  data.agents.forEach((a, i) => {
    const ang = -Math.PI / 2 + i * 2 * Math.PI / n;
    const x = R * Math.cos(ang), z = R * Math.sin(ang);
    S.pos[a.id] = new THREE.Vector3(x, 0, z);
    const strength = { strong: 1, moderate: 0.66, mild: 0.33, none: 0 }[a.profile.strength] || 0;
    const h = 2.0 + strength * 2.6;
    const tx = facade.clone(); tx.repeat.set(1, Math.max(3, Math.round(h * 1.6))); tx.needsUpdate = true;
    const mat = new THREE.MeshStandardMaterial({ color: 0x070c1a, metalness: 0.6, roughness: 0.4,
      emissive: HONEST, emissiveIntensity: 0.7, emissiveMap: tx });
    const tower = new THREE.Mesh(new THREE.BoxGeometry(1.3, h, 1.3), mat);
    tower.position.set(x, h / 2, z);
    tower.userData = { agent: a, h };
    cityGroup.add(tower);
    // a thin roof cap so the top reads as a building crown, not an open box
    const cap = new THREE.Mesh(new THREE.BoxGeometry(1.45, 0.12, 1.45),
      new THREE.MeshStandardMaterial({ color: 0x0a1326, metalness: 0.8, roughness: 0.3,
        emissive: HONEST, emissiveIntensity: 0.4 }));
    cap.position.set(x, h + 0.06, z); cityGroup.add(cap);
    tower.userData.cap = cap;
    S.towers[a.id] = tower;
    const div = document.createElement("div");
    div.className = "tower-label";
    div.textContent = a.id.replace("reviewer_", "R");
    const label = new CSS2DObject(div);
    label.position.set(0, h / 2 + 0.7, 0); tower.add(label);
    tower.userData.label = div;
  });
  // private channels -> arcing conduits between consecutive members
  for (const ch of data.channels) {
    if (ch.id === "public" || ch.members.length < 2) continue;
    for (let i = 0; i < ch.members.length - 1; i++) {
      const a = S.pos[ch.members[i]], b = S.pos[ch.members[i + 1]];
      if (!a || !b) continue;
      const mid = a.clone().add(b).multiplyScalar(0.5); mid.y = 5.0;
      const curve = new THREE.CatmullRomCurve3([a.clone().setY(2.6), mid, b.clone().setY(2.6)]);
      const tube = new THREE.Mesh(new THREE.TubeGeometry(curve, 44, 0.06, 8, false),
        new THREE.MeshBasicMaterial({ color: PRV, transparent: true, opacity: 0.85 }));
      tube.userData = { channel: ch.id, ph: i * 1.3 };
      cityGroup.add(tube); S.conduits.push(tube);
    }
  }
}

function pulse(at, color) {
  const m = new THREE.Mesh(new THREE.SphereGeometry(0.16, 12, 12),
    new THREE.MeshBasicMaterial({ color, transparent: true, opacity: 0.95 }));
  m.position.copy(at);
  S.three.dyn.add(m); S.pulses.push(m);
}

window.Sim3D = {
  scene(data, tick, mode) {
    if (!S.ready && !init()) return;     // 2D fallback handles the no-WebGL case
    S.mode = mode;
    if (S.data !== data) { S.data = data; buildCity(data); }
    const glass = mode === "glassbox";
    // tower facade colour + conduit visibility per observer mode
    for (const a of data.agents) {
      const t = S.towers[a.id]; if (!t) continue;
      const reveal = glass && a.profile.biased;
      const col = reveal ? (DIAL[a.profile.dial] || 0xff6b3d) : HONEST;
      t.material.emissive.setHex(col);
      t.material.emissiveIntensity = reveal ? 1.05 : 0.7;
      if (t.userData.cap) { t.userData.cap.material.emissive.setHex(col); t.userData.cap.material.emissiveIntensity = reveal ? 0.7 : 0.35; }
      if (t.userData.label) t.userData.label.style.color = reveal ? "#ff8fb0" : "#8aa0c8";
    }
    for (const c of S.conduits) c.visible = glass;
    // per-tick events
    S.three.dyn.clear(); S.pulses.length = 0; S.beams.length = 0;
    const tk = data.ticks.find(t => t.tick === tick) || { events: [] };
    for (const e of tk.events) {
      const p = S.pos[e.agent]; if (!p) continue;
      if (e.type === "Speak" && e.ledger === "L_pub") pulse(p.clone().setY(2.4), 0x3d8bff);
      else if (e.type === "Speak" && e.ledger === "L_prv" && glass) pulse(p.clone().setY(4.2), PRV);
      else if (e.type === "DeclareIntent" && glass) pulse(p.clone().setY(4.8), 0xf2c14e);
      else if (e.type === "Vote") {
        const col = e.vote === "accept" ? ACCEPT : REJECT;
        const beam = new THREE.Mesh(new THREE.CylinderGeometry(0.045, 0.045, 7, 8),
          new THREE.MeshBasicMaterial({ color: col, transparent: true, opacity: 0.7 }));
        beam.position.copy(p.clone().setY(5.5)); beam.userData = { ph: Math.random() * 6 };
        S.three.dyn.add(beam); S.beams.push(beam);
      }
    }
  },
};
