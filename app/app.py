from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd

app = Flask(__name__ )

# Load model payload on startup
payload = joblib.load('/home/mo/Desktop/electrical_fault_detection/models/electrical_fault_classifier.joblib')
model = payload['model']
expected_features = payload['features']
target_names = payload['targets']


INDEX_HTML=r"""<!DOCTYPE html>
<html lang="en" class="h-full">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0, viewport-fit=cover">
    <title>Smart Grid — Electrical Line Fault Analysis</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;600;700&display=swap" rel="stylesheet">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    <script>
        tailwind.config = {
            theme: {
                extend: {
                    fontFamily: {
                        sans: ['Inter', 'sans-serif'],
                        display: ['Space Grotesk', 'sans-serif'],
                        mono: ['JetBrains Mono', 'monospace'],
                    }
                }
            }
        }
    </script>
    <style>
        :root {
            --bg: #04060d;
            --panel: rgba(12, 18, 32, 0.68);
            --stroke: rgba(94, 234, 212, 0.14);
            --arc: #5eead4;      /* electric teal — live current */
            --plasma: #c084fc;   /* violet-magenta — engineered feature accent */
            --warn: #fb923c;     /* modern orange — elevated fault */
            --danger: #fb7185;   /* rose — ground fault */
        }
        * { box-sizing: border-box; }
        html, body { background: var(--bg); }
        body { font-family: 'Inter', sans-serif; color: #dce6f2; overflow-x: hidden; }
        h1, h2, h3, .font-display { font-family: 'Space Grotesk', sans-serif; }

        /* ---------- Animated background : circuit mesh + live current ---------- */
        .bg-grid {
            position: fixed; inset: 0; z-index: -3; pointer-events: none;
            background-image:
                linear-gradient(rgba(94,234,212,0.05) 1px, transparent 1px),
                linear-gradient(90deg, rgba(94,234,212,0.05) 1px, transparent 1px);
            background-size: 46px 46px;
            mask-image: radial-gradient(ellipse 90% 75% at 50% 15%, black 25%, transparent 100%);
        }
        .bg-glow-1, .bg-glow-2 {
            position: fixed; z-index: -4; pointer-events: none; border-radius: 50%;
            filter: blur(130px); opacity: .38;
        }
        .bg-glow-1 { width: 640px; height: 640px; top: -230px; left: -150px; background: radial-gradient(circle, #0f766e, transparent 70%); animation: drift 15s ease-in-out infinite alternate; }
        .bg-glow-2 { width: 560px; height: 560px; bottom: -210px; right: -130px; background: radial-gradient(circle, #6d28d9, transparent 70%); animation: drift 19s ease-in-out infinite alternate-reverse; }
        @keyframes drift { from { transform: translate(0,0) scale(1); } to { transform: translate(42px, 32px) scale(1.08); } }

        /* faint circuit-board tracery sweeping behind everything */
        .bg-circuit {
            position: fixed; inset: 0; z-index: -3; pointer-events: none; opacity: .55;
            mask-image: radial-gradient(ellipse 100% 90% at 50% 0%, black 20%, transparent 85%);
        }
        .circuit-trace { stroke: rgba(94,234,212,0.16); stroke-width: 1.2; fill: none; }
        .circuit-node { fill: rgba(94,234,212,0.22); }
        .circuit-sweep {
            stroke: url(#sweepGrad); stroke-width: 1.6; fill: none;
            stroke-dasharray: 6 900; animation: sweep 7s linear infinite;
        }
        @keyframes sweep { to { stroke-dashoffset: -906; } }

        /* flowing current pulses along transmission wires */
        .wire { stroke: rgba(148, 197, 253, 0.22); stroke-width: 2; fill: none; }
        .wire-current {
            stroke: url(#currentGrad); stroke-width: 2.5; fill: none;
            stroke-linecap: round;
            stroke-dasharray: 34 480;
            animation: flow 3s linear infinite;
            filter: drop-shadow(0 0 7px rgba(94, 234, 212, 0.9));
        }
        .wire-current.slow { animation-duration: 4.4s; animation-delay: -1.4s; }
        .wire-current.fast { animation-duration: 2.3s; animation-delay: -0.7s; }
        @keyframes flow { to { stroke-dashoffset: -514; } }
        .electron { filter: drop-shadow(0 0 5px rgba(192,132,252,0.95)); }
        .bg-towers { position: fixed; inset: auto 0 0 0; z-index: -2; pointer-events: none; opacity: .55; }

        /* ---------- Cards ---------- */
        .glass-card {
            background: var(--panel);
            backdrop-filter: blur(16px);
            border: 1px solid var(--stroke);
            border-radius: 1.1rem;
            box-shadow: 0 20px 44px -20px rgba(0,0,0,0.65);
            transition: border-color .2s ease;
        }
        .glass-card:hover { border-color: rgba(94,234,212,0.3); }
        .card-title {
            display: flex; align-items: center; gap: .5rem;
            font-size: .78rem; font-weight: 600; letter-spacing: .06em;
            color: #7dd3c5;
        }

        /* ---------- Dials ---------- */
        .dial-track { stroke: rgba(148,163,184,0.12); stroke-width: 8; fill: none; }
        .dial-progress { stroke-width: 8; stroke-linecap: round; fill: none; transition: stroke-dashoffset .12s ease; }

        input[type=range] { -webkit-appearance: none; appearance: none; height: 6px; border-radius: 999px; background: linear-gradient(90deg, rgba(94,234,212,.35), rgba(192,132,252,.35)); outline: none; }
        input[type=range]::-webkit-slider-thumb {
            -webkit-appearance: none; appearance: none; width: 16px; height: 16px; border-radius: 50%;
            background: #ecfffb; border: 3px solid #0f766e; cursor: pointer;
            box-shadow: 0 0 10px rgba(94,234,212,.85);
        }
        input[type=range]::-moz-range-thumb {
            width: 16px; height: 16px; border-radius: 50%;
            background: #ecfffb; border: 3px solid #0f766e; cursor: pointer;
            box-shadow: 0 0 10px rgba(94,234,212,.85);
        }

        #predict-btn {
            position: relative; overflow: hidden;
            background: linear-gradient(135deg, #0d9488, #7c3aed);
            transition: all .2s ease;
        }
        #predict-btn::after {
            content: ''; position: absolute; inset: 0;
            background: linear-gradient(110deg, transparent 30%, rgba(255,255,255,.22) 50%, transparent 70%);
            transform: translateX(-100%);
        }
        #predict-btn:hover::after { transform: translateX(100%); transition: transform .6s ease; }
        #predict-btn:active { transform: scale(.97); }

        /* ---------- Status / severity colors ---------- */
        .bit-box { transition: all .25s ease; }
        .bit-on { transform: scale(1.06); }
        .blink { animation: blink 1.1s ease-in-out infinite; }
        @keyframes blink { 0%,100% { opacity: 1; } 50% { opacity: .45; } }

        .pulse-dot { animation: pulseDot 1.6s ease-in-out infinite; }
        @keyframes pulseDot { 0%,100% { box-shadow: 0 0 0 0 rgba(94,234,212,.5); } 60% { box-shadow: 0 0 0 8px rgba(94,234,212,0); } }
    </style>
<base target="_blank">
</head>
<body class="min-h-screen antialiased pb-14">

    <!-- ======= BACKGROUND FX : glows, circuit tracery, towers + flowing current ======= -->
    <div class="bg-glow-1"></div>
    <div class="bg-glow-2"></div>
    <div class="bg-grid"></div>

    <svg class="bg-circuit" viewBox="0 0 1440 900" preserveAspectRatio="xMidYMin slice" aria-hidden="true">
        <defs>
            <linearGradient id="sweepGrad" x1="0" y1="0" x2="1" y2="0">
                <stop offset="0%" stop-color="#5eead4" stop-opacity="0"/>
                <stop offset="50%" stop-color="#99f6e4"/>
                <stop offset="100%" stop-color="#c084fc" stop-opacity="0"/>
            </linearGradient>
        </defs>
        <g class="circuit-trace">
            <path d="M60,40 H340 V180 H620 V60 H980 V220 H1380"/>
            <path d="M120,300 H460 V420 H820 V320 H1160 V480 H1400"/>
            <path d="M40,560 H300 V680 H700 V600 H1040 V760 H1360"/>
        </g>
        <g class="circuit-node">
            <circle cx="340" cy="40" r="3"/><circle cx="620" cy="180" r="3"/><circle cx="980" cy="60" r="3"/><circle cx="1380" cy="220" r="3"/>
            <circle cx="460" cy="300" r="3"/><circle cx="820" cy="420" r="3"/><circle cx="1160" cy="320" r="3"/>
            <circle cx="300" cy="560" r="3"/><circle cx="700" cy="680" r="3"/><circle cx="1040" cy="600" r="3"/>
        </g>
        <path class="circuit-sweep" d="M60,40 H340 V180 H620 V60 H980 V220 H1380"/>
        <path class="circuit-sweep" style="animation-delay:-3.5s" d="M120,300 H460 V420 H820 V320 H1160 V480 H1400"/>
    </svg>

    <svg class="bg-towers" viewBox="0 0 1440 420" preserveAspectRatio="xMidYMax slice" aria-hidden="true">
        <defs>
            <linearGradient id="currentGrad" x1="0" y1="0" x2="1" y2="0">
                <stop offset="0%" stop-color="#5eead4" stop-opacity="0"/>
                <stop offset="50%" stop-color="#99f6e4"/>
                <stop offset="100%" stop-color="#c084fc" stop-opacity="0"/>
            </linearGradient>
        </defs>

        <!-- catenary wires -->
        <path id="wireA" class="wire" d="M0,110 Q360,210 720,130 T1440,110"/>
        <path id="wireB" class="wire" d="M0,150 Q360,250 720,170 T1440,150"/>
        <path id="wireC" class="wire" d="M0,190 Q360,290 720,210 T1440,190"/>

        <!-- flowing current pulses -->
        <path class="wire-current"      d="M0,110 Q360,210 720,130 T1440,110"/>
        <path class="wire-current slow" d="M0,150 Q360,250 720,170 T1440,150"/>
        <path class="wire-current fast" d="M0,190 Q360,290 720,210 T1440,190"/>

        <!-- traveling electron particles riding the wires -->
        <circle r="3" class="electron" fill="#99f6e4">
            <animateMotion dur="3.4s" repeatCount="indefinite"><mpath href="#wireA"/></animateMotion>
        </circle>
        <circle r="2.6" class="electron" fill="#c084fc">
            <animateMotion dur="4.8s" repeatCount="indefinite" begin="-1.2s"><mpath href="#wireB"/></animateMotion>
        </circle>
        <circle r="2.6" class="electron" fill="#67e8f9">
            <animateMotion dur="2.6s" repeatCount="indefinite" begin="-0.6s"><mpath href="#wireC"/></animateMotion>
        </circle>

        <!-- transmission towers -->
        <g stroke="rgba(125,170,200,0.5)" stroke-width="3" fill="none" stroke-linecap="round">
            <!-- Tower 1 -->
            <g transform="translate(240,70)">
                <path d="M-30,330 L0,0 L30,330"/>
                <path d="M-22,240 L22,240 M-16,165 L16,165 M-9,85 L9,85"/>
                <path d="M-105,55 L105,55 M-85,20 L85,20"/>
                <path d="M-105,55 L-105,90 M105,55 L105,90 M-85,20 L-85,55 M85,20 L85,55"/>
                <circle cx="-105" cy="94" r="4" fill="rgba(94,234,212,.85)" stroke="none"/>
                <circle cx="105" cy="94" r="4" fill="rgba(94,234,212,.85)" stroke="none"/>
                <circle cx="-85" cy="59" r="4" fill="rgba(94,234,212,.85)" stroke="none"/>
                <circle cx="85" cy="59" r="4" fill="rgba(94,234,212,.85)" stroke="none"/>
            </g>
            <!-- Tower 2 -->
            <g transform="translate(720,55) scale(1.15)">
                <path d="M-30,330 L0,0 L30,330"/>
                <path d="M-22,240 L22,240 M-16,165 L16,165 M-9,85 L9,85"/>
                <path d="M-105,55 L105,55 M-85,20 L85,20"/>
                <path d="M-105,55 L-105,90 M105,55 L105,90 M-85,20 L-85,55 M85,20 L85,55"/>
                <circle cx="-105" cy="94" r="4" fill="rgba(192,132,252,.85)" stroke="none"/>
                <circle cx="105" cy="94" r="4" fill="rgba(192,132,252,.85)" stroke="none"/>
                <circle cx="-85" cy="59" r="4" fill="rgba(192,132,252,.85)" stroke="none"/>
                <circle cx="85" cy="59" r="4" fill="rgba(192,132,252,.85)" stroke="none"/>
            </g>
            <!-- Tower 3 -->
            <g transform="translate(1190,70)">
                <path d="M-30,330 L0,0 L30,330"/>
                <path d="M-22,240 L22,240 M-16,165 L16,165 M-9,85 L9,85"/>
                <path d="M-105,55 L105,55 M-85,20 L85,20"/>
                <path d="M-105,55 L-105,90 M105,55 L105,90 M-85,20 L-85,55 M85,20 L85,55"/>
                <circle cx="-105" cy="94" r="4" fill="rgba(94,234,212,.85)" stroke="none"/>
                <circle cx="105" cy="94" r="4" fill="rgba(94,234,212,.85)" stroke="none"/>
                <circle cx="-85" cy="59" r="4" fill="rgba(94,234,212,.85)" stroke="none"/>
                <circle cx="85" cy="59" r="4" fill="rgba(94,234,212,.85)" stroke="none"/>
            </g>
        </g>
    </svg>

    <!-- ======= HEADER ======= -->
    <header class="sticky top-0 z-50 border-b border-slate-800/70 bg-[#04060d]/80 backdrop-blur-xl">
        <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-18 py-3 flex items-center justify-between gap-4">
            <div class="flex items-center gap-4">
                <!-- transmission line logo -->
                <svg width="52" height="52" viewBox="0 0 64 64" aria-hidden="true">
                    <defs>
                        <linearGradient id="logoGrad" x1="0" y1="0" x2="1" y2="1">
                            <stop offset="0%" stop-color="#5eead4"/>
                            <stop offset="100%" stop-color="#a855f7"/>
                        </linearGradient>
                    </defs>
                    <rect x="2" y="2" width="60" height="60" rx="14" fill="rgba(8,20,40,.9)" stroke="url(#logoGrad)" stroke-width="2"/>
                    <g stroke="url(#logoGrad)" stroke-width="2.6" fill="none" stroke-linecap="round">
                        <path d="M24,52 L32,14 L40,52"/>
                        <path d="M26.5,40 L37.5,40 M29,28 L35,28"/>
                        <path d="M14,24 L50,24"/>
                        <path d="M14,24 L14,31 M50,24 L50,31"/>
                    </g>
                    <path d="M33.5,31 L29,39 L32,39 L30.5,47 L36,37.5 L33,37.5 L35.5,31 Z" fill="#99f6e4"/>
                    <circle cx="14" cy="33" r="2.4" fill="#99f6e4"/>
                    <circle cx="50" cy="33" r="2.4" fill="#99f6e4"/>
                </svg>
                <div>
                    <h1 class="text-lg sm:text-xl font-bold text-white tracking-tight leading-tight font-display">
                        Smart Grid <span class="text-transparent bg-clip-text bg-gradient-to-r from-teal-300 to-purple-300">Electrical Line Fault Analysis</span>
                    </h1>
                    <p class="text-[11px] text-slate-400 tracking-wide">Random Forest high-voltage diagnostics system</p>
                </div>
            </div>
            <div class="hidden sm:flex items-center gap-3">
                <span id="model-status-badge" class="inline-flex items-center px-3 py-1 rounded-full text-[11px] font-semibold bg-slate-700/40 text-slate-400 border border-slate-600">
                    <span class="w-2 h-2 rounded-full bg-slate-500 mr-2"></span>
                    Awaiting first inference
                </span>
            </div>
        </div>
    </header>

    <main class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 pt-8 space-y-8">

        <!-- ======= MODEL OVERVIEW ======= -->
        <section class="glass-card p-6 sm:p-8">
            <div class="flex flex-col md:flex-row gap-6 items-start justify-between">
                <div class="space-y-3 max-w-3xl">
                    <div class="card-title"><i class="fa-solid fa-circle-info"></i><span>System architecture &amp; problem statement</span></div>
                    <h2 class="text-2xl font-bold text-white font-display">Why this model was engineered</h2>
                    <p class="text-slate-300 text-sm leading-relaxed">
                        High-voltage transmission lines are vulnerable to environmental hazards, lightning strikes, insulation breakdown, and aging infrastructure. Raw 3-phase current (I<sub>a</sub>, I<sub>b</sub>, I<sub>c</sub>) and voltage (V<sub>a</sub>, V<sub>b</sub>, V<sub>c</sub>) measurements often exhibit <strong class="text-white">class overlap</strong> in symmetrical and asymmetrical fault states, causing standard classifiers to mispredict ground interactions.
                    </p>
                    <p class="text-slate-300 text-sm leading-relaxed">
                        To resolve this, the pipeline dynamically computes <strong class="text-teal-300">zero-sequence components</strong>:
                        <code class="bg-slate-800/80 px-1.5 py-0.5 rounded text-teal-300 font-mono text-xs">I₀ = Iₐ + I_b + I_c</code> and
                        <code class="bg-slate-800/80 px-1.5 py-0.5 rounded text-teal-300 font-mono text-xs">V₀ = Vₐ + V_b + V_c</code>.
                        Including I₀ and V₀ unmasks unbalance and ground leakage instantly, enabling the Random Forest classifier to reach <strong class="text-white">100% precision fault detection</strong> across all target channels (G, C, B, A).
                    </p>
                </div>
                <div class="bg-slate-900/60 rounded-xl p-4 border border-slate-800 space-y-3 w-full md:w-72 shrink-0">
                    <div class="text-[11px] font-semibold text-slate-400 tracking-wider">Target channels</div>
                    <div class="grid grid-cols-2 gap-2 text-center font-mono">
                        <div class="bg-slate-800/60 p-2.5 rounded-lg border border-slate-700/70">
                            <span class="block text-slate-500 text-[10px]">Ground</span>
                            <span class="text-lg font-bold text-orange-400">G</span>
                        </div>
                        <div class="bg-slate-800/60 p-2.5 rounded-lg border border-slate-700/70">
                            <span class="block text-slate-500 text-[10px]">Phase C</span>
                            <span class="text-lg font-bold text-sky-400">C</span>
                        </div>
                        <div class="bg-slate-800/60 p-2.5 rounded-lg border border-slate-700/70">
                            <span class="block text-slate-500 text-[10px]">Phase B</span>
                            <span class="text-lg font-bold text-purple-400">B</span>
                        </div>
                        <div class="bg-slate-800/60 p-2.5 rounded-lg border border-slate-700/70">
                            <span class="block text-slate-500 text-[10px]">Phase A</span>
                            <span class="text-lg font-bold text-rose-400">A</span>
                        </div>
                    </div>
                </div>
            </div>
        </section>

        <!-- ======= MAIN GRID ======= -->
        <section class="grid grid-cols-1 lg:grid-cols-12 gap-6">

            <!-- LEFT (8 cols): dials -->
            <div class="lg:col-span-8 space-y-6">

                <!-- 3-Phase Currents -->
                <div class="glass-card p-6">
                    <div class="flex items-center justify-between mb-5">
                        <h3 class="card-title"><i class="fa-solid fa-wave-square text-teal-300"></i> 3-phase line currents <span class="text-slate-500 font-normal">(±1000 A)</span></h3>
                        <span class="text-[11px] text-slate-500">Adjust sliders or type directly</span>
                    </div>

                    <div class="grid grid-cols-1 sm:grid-cols-3 gap-5">
                        <!-- Ia -->
                        <div class="bg-slate-900/50 p-4 rounded-xl border border-slate-800/70 flex flex-col items-center">
                            <span class="text-[11px] font-bold text-rose-400 tracking-wider mb-3">Phase A current (Iₐ)</span>
                            <div class="relative w-32 h-32 flex items-center justify-center">
                                <svg class="w-full h-full -rotate-90" viewBox="0 0 100 100">
                                    <circle class="dial-track" cx="50" cy="50" r="40"></circle>
                                    <circle id="dial-Ia" class="dial-progress stroke-rose-400" cx="50" cy="50" r="40" stroke-dasharray="251.2" stroke-dashoffset="125.6"></circle>
                                </svg>
                                <div class="absolute text-center">
                                    <input type="number" id="num-Ia" min="-1000" max="1000" value="-882" oninput="syncFromInput('Ia', this.value)" class="w-20 bg-transparent text-center font-mono font-bold text-lg text-white focus:outline-none border-b border-transparent focus:border-rose-400">
                                    <span class="block text-[10px] text-slate-500 font-semibold">amperes</span>
                                </div>
                            </div>
                            <input type="range" id="range-Ia" min="-1000" max="1000" step="1" value="-882" oninput="updateDial('Ia', this.value)" class="w-full mt-4 cursor-pointer">
                        </div>
                        <!-- Ib -->
                        <div class="bg-slate-900/50 p-4 rounded-xl border border-slate-800/70 flex flex-col items-center">
                            <span class="text-[11px] font-bold text-purple-400 tracking-wider mb-3">Phase B current (I_b)</span>
                            <div class="relative w-32 h-32 flex items-center justify-center">
                                <svg class="w-full h-full -rotate-90" viewBox="0 0 100 100">
                                    <circle class="dial-track" cx="50" cy="50" r="40"></circle>
                                    <circle id="dial-Ib" class="dial-progress stroke-purple-400" cx="50" cy="50" r="40" stroke-dasharray="251.2" stroke-dashoffset="125.6"></circle>
                                </svg>
                                <div class="absolute text-center">
                                    <input type="number" id="num-Ib" min="-1000" max="1000" value="884" oninput="syncFromInput('Ib', this.value)" class="w-20 bg-transparent text-center font-mono font-bold text-lg text-white focus:outline-none border-b border-transparent focus:border-purple-400">
                                    <span class="block text-[10px] text-slate-500 font-semibold">amperes</span>
                                </div>
                            </div>
                            <input type="range" id="range-Ib" min="-1000" max="1000" step="1" value="884" oninput="updateDial('Ib', this.value)" class="w-full mt-4 cursor-pointer">
                        </div>
                        <!-- Ic -->
                        <div class="bg-slate-900/50 p-4 rounded-xl border border-slate-800/70 flex flex-col items-center">
                            <span class="text-[11px] font-bold text-sky-400 tracking-wider mb-3">Phase C current (I_c)</span>
                            <div class="relative w-32 h-32 flex items-center justify-center">
                                <svg class="w-full h-full -rotate-90" viewBox="0 0 100 100">
                                    <circle class="dial-track" cx="50" cy="50" r="40"></circle>
                                    <circle id="dial-Ic" class="dial-progress stroke-sky-400" cx="50" cy="50" r="40" stroke-dasharray="251.2" stroke-dashoffset="125.6"></circle>
                                </svg>
                                <div class="absolute text-center">
                                    <input type="number" id="num-Ic" min="-1000" max="1000" value="-1.2" step="0.1" oninput="syncFromInput('Ic', this.value)" class="w-20 bg-transparent text-center font-mono font-bold text-lg text-white focus:outline-none border-b border-transparent focus:border-sky-400">
                                    <span class="block text-[10px] text-slate-500 font-semibold">amperes</span>
                                </div>
                            </div>
                            <input type="range" id="range-Ic" min="-1000" max="1000" step="1" value="-1" oninput="updateDial('Ic', this.value)" class="w-full mt-4 cursor-pointer">
                        </div>
                    </div>
                </div>

                <!-- 3-Phase Voltages -->
                <div class="glass-card p-6">
                    <div class="flex items-center justify-between mb-5">
                        <h3 class="card-title"><i class="fa-solid fa-car-battery text-orange-300"></i> 3-phase line voltages <span class="text-slate-500 font-normal">(±1.0 p.u.)</span></h3>
                        <span class="text-[11px] text-slate-500">Normalized per-unit magnitude</span>
                    </div>

                    <div class="grid grid-cols-1 sm:grid-cols-3 gap-5">
                        <!-- Va -->
                        <div class="bg-slate-900/50 p-4 rounded-xl border border-slate-800/70 flex flex-col items-center">
                            <span class="text-[11px] font-bold text-rose-400 tracking-wider mb-3">Phase A voltage (Vₐ)</span>
                            <div class="relative w-32 h-32 flex items-center justify-center">
                                <svg class="w-full h-full -rotate-90" viewBox="0 0 100 100">
                                    <circle class="dial-track" cx="50" cy="50" r="40"></circle>
                                    <circle id="dial-Va" class="dial-progress stroke-rose-300" cx="50" cy="50" r="40" stroke-dasharray="251.2" stroke-dashoffset="125.6"></circle>
                                </svg>
                                <div class="absolute text-center">
                                    <input type="number" id="num-Va" min="-1.0" max="1.0" step="0.01" value="-0.62" oninput="syncFromInput('Va', this.value)" class="w-20 bg-transparent text-center font-mono font-bold text-lg text-white focus:outline-none border-b border-transparent focus:border-rose-300">
                                    <span class="block text-[10px] text-slate-500 font-semibold">p.u.</span>
                                </div>
                            </div>
                            <input type="range" id="range-Va" min="-1.0" max="1.0" step="0.01" value="-0.62" oninput="updateDial('Va', this.value)" class="w-full mt-4 cursor-pointer">
                        </div>
                        <!-- Vb -->
                        <div class="bg-slate-900/50 p-4 rounded-xl border border-slate-800/70 flex flex-col items-center">
                            <span class="text-[11px] font-bold text-purple-400 tracking-wider mb-3">Phase B voltage (V_b)</span>
                            <div class="relative w-32 h-32 flex items-center justify-center">
                                <svg class="w-full h-full -rotate-90" viewBox="0 0 100 100">
                                    <circle class="dial-track" cx="50" cy="50" r="40"></circle>
                                    <circle id="dial-Vb" class="dial-progress stroke-purple-300" cx="50" cy="50" r="40" stroke-dasharray="251.2" stroke-dashoffset="125.6"></circle>
                                </svg>
                                <div class="absolute text-center">
                                    <input type="number" id="num-Vb" min="-1.0" max="1.0" step="0.01" value="0.62" oninput="syncFromInput('Vb', this.value)" class="w-20 bg-transparent text-center font-mono font-bold text-lg text-white focus:outline-none border-b border-transparent focus:border-purple-300">
                                    <span class="block text-[10px] text-slate-500 font-semibold">p.u.</span>
                                </div>
                            </div>
                            <input type="range" id="range-Vb" min="-1.0" max="1.0" step="0.01" value="0.62" oninput="updateDial('Vb', this.value)" class="w-full mt-4 cursor-pointer">
                        </div>
                        <!-- Vc -->
                        <div class="bg-slate-900/50 p-4 rounded-xl border border-slate-800/70 flex flex-col items-center">
                            <span class="text-[11px] font-bold text-sky-400 tracking-wider mb-3">Phase C voltage (V_c)</span>
                            <div class="relative w-32 h-32 flex items-center justify-center">
                                <svg class="w-full h-full -rotate-90" viewBox="0 0 100 100">
                                    <circle class="dial-track" cx="50" cy="50" r="40"></circle>
                                    <circle id="dial-Vc" class="dial-progress stroke-sky-300" cx="50" cy="50" r="40" stroke-dasharray="251.2" stroke-dashoffset="125.6"></circle>
                                </svg>
                                <div class="absolute text-center">
                                    <input type="number" id="num-Vc" min="-1.0" max="1.0" step="0.01" value="0.00" oninput="syncFromInput('Vc', this.value)" class="w-20 bg-transparent text-center font-mono font-bold text-lg text-white focus:outline-none border-b border-transparent focus:border-sky-300">
                                    <span class="block text-[10px] text-slate-500 font-semibold">p.u.</span>
                                </div>
                            </div>
                            <input type="range" id="range-Vc" min="-1.0" max="1.0" step="0.01" value="0.00" oninput="updateDial('Vc', this.value)" class="w-full mt-4 cursor-pointer">
                        </div>
                    </div>
                </div>

            </div>

            <!-- RIGHT (4 cols): zero-sequence + inference + result -->
            <div class="lg:col-span-4 space-y-6">

                <!-- Zero-sequence calculations -->
                <div class="glass-card p-6 space-y-4">
                    <div class="flex items-center justify-between border-b border-slate-800 pb-3">
                        <h3 class="card-title"><i class="fa-solid fa-calculator text-teal-300"></i> Engineered zero-sequence</h3>
                        <span class="text-[10px] bg-teal-500/10 text-teal-300 border border-teal-500/30 px-2 py-0.5 rounded font-mono">dynamic</span>
                    </div>

                    <div class="bg-slate-900/60 rounded-xl p-4 border border-slate-800/70">
                        <div class="flex justify-between items-center text-[11px] mb-1">
                            <span class="text-slate-400 font-semibold">Zero-sequence current</span>
                            <span class="font-mono text-slate-500">I₀ = Iₐ + I_b + I_c</span>
                        </div>
                        <div class="flex items-baseline justify-between">
                            <span id="disp-I0" class="text-2xl font-mono font-bold text-teal-300">0.80</span>
                            <span class="text-[11px] text-slate-500 font-semibold">amperes</span>
                        </div>
                    </div>

                    <div class="bg-slate-900/60 rounded-xl p-4 border border-slate-800/70">
                        <div class="flex justify-between items-center text-[11px] mb-1">
                            <span class="text-slate-400 font-semibold">Zero-sequence voltage</span>
                            <span class="font-mono text-slate-500">V₀ = Vₐ + V_b + V_c</span>
                        </div>
                        <div class="flex items-baseline justify-between">
                            <span id="disp-V0" class="text-2xl font-mono font-bold text-teal-300">0.00</span>
                            <span class="text-[11px] text-slate-500 font-semibold">p.u.</span>
                        </div>
                    </div>

                    <button onclick="runInference()" id="predict-btn" class="w-full py-3 px-4 rounded-xl text-white font-bold text-sm shadow-lg shadow-teal-900/40 flex items-center justify-center gap-2">
                        <i class="fa-solid fa-microchip text-base"></i>
                        <span>Execute model inference</span>
                    </button>
                    <p class="text-[10px] text-slate-500 text-center -mt-1">Presets assign inputs only — push the button to run inference.</p>
                </div>

                <!-- Diagnostic result -->
                <div class="glass-card p-6 space-y-5">
                    <div class="flex items-center justify-between border-b border-slate-800 pb-3">
                        <h3 class="card-title"><i class="fa-solid fa-triangle-exclamation text-orange-300"></i> Diagnostic result</h3>
                        <span id="status-badge" class="text-[11px] font-bold px-2.5 py-0.5 rounded bg-slate-700/60 text-slate-300 border border-slate-600">awaiting inference</span>
                    </div>

                    <!-- Diagnostic banner -->
                    <div id="fault-banner" class="p-4 rounded-xl bg-slate-900/60 border border-slate-800/70 transition-all">
                        <span class="text-[10px] font-bold text-slate-500 block mb-1">Mapped fault classification</span>
                        <div id="fault-title" class="text-base font-bold text-slate-400 flex items-center gap-2">
                            <i class="fa-solid fa-hourglass-half"></i> Awaiting model inference
                        </div>
                        <p id="fault-desc" class="text-xs text-slate-500 mt-1">Set the phase signals, then press "Execute model inference".</p>
                    </div>
                </div>

            </div>
        </section>

        <!-- ======= FAULT LOOKUP MATRIX ======= -->
        <section class="glass-card p-6">
            <h3 class="text-sm font-bold text-white mb-4 flex items-center gap-2 font-display">
                <i class="fa-solid fa-book-open text-teal-300"></i>
                <span class="card-title">Fault code mapping matrix reference</span>
            </h3>
            <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3 text-xs">
                <div class="bg-slate-900/60 p-3 rounded-lg border border-slate-800 flex items-center justify-between"><span class="font-mono text-teal-300 font-bold">[0 0 0 0]</span><span class="text-slate-300">No fault (normal grid)</span></div>
                <div class="bg-slate-900/60 p-3 rounded-lg border border-slate-800 flex items-center justify-between"><span class="font-mono text-rose-400 font-bold">[1 0 0 1]</span><span class="text-slate-300">LG fault (phase A to gnd)</span></div>
                <div class="bg-slate-900/60 p-3 rounded-lg border border-slate-800 flex items-center justify-between"><span class="font-mono text-rose-400 font-bold">[1 0 1 0]</span><span class="text-slate-300">LG fault (phase B to gnd)</span></div>
                <div class="bg-slate-900/60 p-3 rounded-lg border border-slate-800 flex items-center justify-between"><span class="font-mono text-rose-400 font-bold">[1 1 0 0]</span><span class="text-slate-300">LG fault (phase C to gnd)</span></div>
                <div class="bg-slate-900/60 p-3 rounded-lg border border-slate-800 flex items-center justify-between"><span class="font-mono text-fuchsia-400 font-bold">[0 0 1 1]</span><span class="text-slate-300">LL fault (phase A to phase B)</span></div>
                <div class="bg-slate-900/60 p-3 rounded-lg border border-slate-800 flex items-center justify-between"><span class="font-mono text-fuchsia-400 font-bold">[0 1 1 0]</span><span class="text-slate-300">LL fault (phase B to phase C)</span></div>
                <div class="bg-slate-900/60 p-3 rounded-lg border border-slate-800 flex items-center justify-between"><span class="font-mono text-orange-400 font-bold">[1 0 1 1]</span><span class="text-slate-300">LLG fault (phases A,B to gnd)</span></div>
                <div class="bg-slate-900/60 p-3 rounded-lg border border-slate-800 flex items-center justify-between"><span class="font-mono text-purple-400 font-bold">[0 1 1 1]</span><span class="text-slate-300">LLL fault (phases A,B,C short)</span></div>
                <div class="bg-slate-900/60 p-3 rounded-lg border border-slate-800 flex items-center justify-between"><span class="font-mono text-red-500 font-bold">[1 1 1 1]</span><span class="text-slate-300">LLLG fault (symmetrical ground)</span></div>
            </div>
        </section>

    </main>

    <script>
    /* ================= INITIAL STATE (Fixed to Normal Balanced Grid) ================= */
    const signals = {
        Ia: 200.0,  Ib: -100.0, Ic: -100.0,
        Va: 0.80,   Vb: -0.40,  Vc: -0.40,
        I0: 0.0,    V0: 0.0
    };

    const limits = {
        Ia: { min: -1000, max: 1000 },
        Ib: { min: -1000, max: 1000 },
        Ic: { min: -1000, max: 1000 },
        Va: { min: -1.0, max: 1.0 },
        Vb: { min: -1.0, max: 1.0 },
        Vc: { min: -1.0, max: 1.0 }
    };

    /* ================= INIT ================= */
    window.addEventListener('DOMContentLoaded', () => {
        Object.keys(limits).forEach(key => updateDialUI(key, signals[key]));
        recalculateSequenceComponents();
    });

    /* ================= DIAL UI ================= */
    function updateDialUI(key, val) {
        if (!limits[key]) return;
        const range = limits[key];
        const clamped = Math.max(range.min, Math.min(range.max, parseFloat(val) || 0));
        signals[key] = clamped;

        const ratio = (clamped - range.min) / (range.max - range.min);
        const circumference = 2 * Math.PI * 40;
        const offset = circumference * (1 - ratio);

        const dialArc = document.getElementById(`dial-${key}`);
        if (dialArc) dialArc.style.strokeDashoffset = offset;

        const numInput = document.getElementById(`num-${key}`);
        const rangeInput = document.getElementById(`range-${key}`);
        if (numInput && document.activeElement !== numInput) numInput.value = clamped;
        if (rangeInput && document.activeElement !== rangeInput) rangeInput.value = clamped;
    }

    function updateDial(key, val) {
        updateDialUI(key, val);
        recalculateSequenceComponents();
    }

    function syncFromInput(key, val) {
        updateDialUI(key, val);
        recalculateSequenceComponents();
    }

    /* ================= DYNAMIC ZERO-SEQUENCE (Fix: Save directly to signals) ================= */
    function recalculateSequenceComponents() {
        signals.I0 = Number((signals.Ia + signals.Ib + signals.Ic).toFixed(2));
        signals.V0 = Number((signals.Va + signals.Vb + signals.Vc).toFixed(2));

        const dispI0 = document.getElementById('disp-I0');
        const dispV0 = document.getElementById('disp-V0');
        if (dispI0) dispI0.innerText = signals.I0.toFixed(2);
        if (dispV0) dispV0.innerText = signals.V0.toFixed(2);
    }

    /* ================= MODEL INFERENCE ================= */
    async function runInference() {
        const btn = document.getElementById('predict-btn');
        if (btn) btn.classList.add('opacity-75', 'cursor-not-allowed', 'pointer-events-none');
        setBadge('inferencing…', 'cyan');

        try {
            // Ensure I0 and V0 are refreshed before send
            recalculateSequenceComponents();

            const response = await fetch('/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(signals) // Now includes all 8 features: Ia, Ib, Ic, Va, Vb, Vc, I0, V0
            });

            if (!response.ok) {
                throw new Error(`Server responded with status ${response.status}`);
            }

            const resData = await response.json();

            if (resData.status === 'success' && resData.inference_result) {
                renderPrediction(resData.inference_result);
                setHeaderStatus(true);
            } else {
                throw new Error(resData.message || 'Model returned an invalid result.');
            }
        } catch (err) {
            console.error('Prediction error:', err);
            renderModelError();
        } finally {
            if (btn) btn.classList.remove('opacity-75', 'cursor-not-allowed', 'pointer-events-none');
        }
    }

    function setHeaderStatus(loaded) {
        const badge = document.getElementById('model-status-badge');
        if (!badge) return;
        if (loaded) {
            badge.className = 'inline-flex items-center px-3 py-1 rounded-full text-[11px] font-semibold bg-teal-500/10 text-teal-300 border border-teal-500/30';
            badge.innerHTML = '<span class="w-2 h-2 rounded-full bg-teal-300 mr-2 pulse-dot"></span>Model payload loaded';
        } else {
            badge.className = 'inline-flex items-center px-3 py-1 rounded-full text-[11px] font-semibold bg-red-500/10 text-red-400 border border-red-500/30';
            badge.innerHTML = '<span class="w-2 h-2 rounded-full bg-red-400 mr-2"></span>Model not loaded';
        }
    }

    function renderModelError() {
        setHeaderStatus(false);
        ['G', 'C', 'B', 'A'].forEach(ch => {
            const el = document.getElementById(`bit-${ch}`);
            if (el) {
                el.className = 'bit-box bg-slate-900 p-2.5 rounded-lg border border-slate-800 text-slate-600';
                const v = el.querySelector('span:nth-child(2)');
                if (v) v.innerText = '–';
            }
        });

        setBadge('model offline', 'red');

        const title = document.getElementById('fault-title');
        const desc  = document.getElementById('fault-desc');
        if (title) {
            title.className = 'text-base font-bold text-red-400 flex items-center gap-2';
            title.innerHTML = '<i class="fa-solid fa-circle-exclamation blink"></i> Prediction unavailable — model not loaded';
        }
        if (desc) {
            desc.className  = 'text-xs text-slate-400 mt-1';
            desc.innerText  = 'The Random Forest model could not be reached or failed to load on the Flask backend. Please verify that app.py is running and the model payload is loaded, then try again.';
        }
    }

    function setBadge(text, color) {
        const badge = document.getElementById('status-badge');
        if (!badge) return;
        const themes = {
            cyan:   'bg-teal-500/15 text-teal-300 border-teal-500/40',
            red:    'bg-red-500/15 text-red-400 border-red-500/40',
            amber:  'bg-orange-500/15 text-orange-400 border-orange-500/40',
            slate:  'bg-slate-700/60 text-slate-300 border-slate-600'
        };
        badge.className = `text-[11px] font-bold px-2.5 py-0.5 rounded border ${themes[color] || themes.slate}`;
        badge.innerText = text;
    }

    /* ================= DECODER MAPPING ================= */
    function renderPrediction(pred) {
        const g = pred.G ?? 0;
        const c = pred.C ?? 0;
        const b = pred.B ?? 0;
        const a = pred.A ?? 0;

        updateBitBox('bit-G', g, 'orange');
        updateBitBox('bit-C', c, 'sky');
        updateBitBox('bit-B', b, 'purple');
        updateBitBox('bit-A', a, 'rose');

        const key = `${g}${c}${b}${a}`;

        const mapping = {
            '0000': { title: 'No fault (normal operation)', desc: 'Balanced 3-phase grid condition. No ground leak or line short.', color: 'teal', badge: 'normal' },
            '1001': { title: 'Line-to-ground fault (A-G)', desc: 'Phase A insulation breakdown to ground.', color: 'rose', badge: 'critical fault' },
            '1010': { title: 'Line-to-ground fault (B-G)', desc: 'Phase B shorted to ground.', color: 'rose', badge: 'critical fault' },
            '1100': { title: 'Line-to-ground fault (C-G)', desc: 'Phase C shorted to ground.', color: 'rose', badge: 'critical fault' },
            '0011': { title: 'Line-to-line fault (A-B)', desc: 'Direct short circuit between Phase A and Phase B.', color: 'fuchsia', badge: 'high fault' },
            '0110': { title: 'Line-to-line fault (B-C)', desc: 'Direct short circuit between Phase B and Phase C.', color: 'fuchsia', badge: 'high fault' },
            '0101': { title: 'Line-to-line fault (A-C)', desc: 'Direct short circuit between Phase A and Phase C.', color: 'fuchsia', badge: 'high fault' },
            '1011': { title: 'Double line-to-ground fault (A-B-G)', desc: 'Phases A and B shorted together and connected to ground.', color: 'amber', badge: 'critical fault' },
            '1110': { title: 'Double line-to-ground fault (B-C-G)', desc: 'Phases B and C shorted together and connected to ground.', color: 'amber', badge: 'critical fault' },
            '1101': { title: 'Double line-to-ground fault (A-C-G)', desc: 'Phases A and C shorted together and connected to ground.', color: 'amber', badge: 'critical fault' },
            '0111': { title: 'Three-phase fault (A-B-C)', desc: 'Symmetrical three-phase short circuit without ground contact.', color: 'purple', badge: 'severe fault' },
            '1111': { title: 'Three-phase symmetrical fault (A-B-C-G)', desc: 'Total line collapse across all 3 phases to ground.', color: 'red', badge: 'extreme fault' }
        };

        const match = mapping[key] || {
            title: `Unmapped fault combination [${g} ${c} ${b} ${a}]`,
            desc: 'Non-standard transient fault vector recorded.',
            color: 'amber',
            badge: 'unknown fault'
        };

        const title = document.getElementById('fault-title');
        const desc  = document.getElementById('fault-desc');

        if (title) title.innerText = match.title;
        if (desc)  desc.innerText  = match.desc;

        if (match.color === 'teal') {
            setBadge('normal', 'cyan');
            if (title) title.className = 'text-base font-bold text-teal-300 flex items-center gap-2';
        } else if (match.color === 'red' || match.color === 'rose') {
            setBadge(match.badge, 'red');
            badgeBlink();
            if (title) title.className = 'text-base font-bold text-rose-400 flex items-center gap-2';
        } else if (match.color === 'fuchsia' || match.color === 'purple') {
            setBadge(match.badge, 'cyan');
            if (title) title.className = 'text-base font-bold text-purple-300 flex items-center gap-2';
        } else {
            setBadge(match.badge, 'amber');
            if (title) title.className = 'text-base font-bold text-orange-400 flex items-center gap-2';
        }
    }

    function badgeBlink() {
        const badge = document.getElementById('status-badge');
        if (!badge) return;
        badge.classList.add('blink');
        setTimeout(() => badge.classList.remove('blink'), 4000);
    }

    /* Fixed dynamic Tailwind CSS class lookup */
    function updateBitBox(id, bitVal, themeColor) {
        const el = document.getElementById(id);
        if (!el) return;

        const themeClasses = {
            orange: 'bg-orange-500/15 border-orange-500/50 text-orange-300',
            sky:    'bg-sky-500/15 border-sky-500/50 text-sky-300',
            purple: 'bg-purple-500/15 border-purple-500/50 text-purple-300',
            rose:   'bg-rose-500/15 border-rose-500/50 text-rose-300'
        };

        const valEl = el.querySelector('span:nth-child(2)');
        if (bitVal === 1) {
            const activeStyle = themeClasses[themeColor] || themeClasses.orange;
            el.className = `bit-box bit-on ${activeStyle} p-2.5 rounded-lg border font-bold`;
            if (valEl) valEl.innerText = '1';
        } else {
            el.className = 'bit-box bg-slate-900 p-2.5 rounded-lg border border-slate-800 text-slate-500';
            if (valEl) valEl.innerText = '0';
        }
    }
</script>
</body>
</html>

"""

# Serve the HTML User Interface
@app.get('/')
def index():
    return INDEX_HTML

# API Inference Endpoint
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        
        # Convert input to DataFrame
        df_sample = pd.DataFrame([data])
        
        # Calculate zero-sequence features
        df_sample['I0'] = df_sample['Ia'] + df_sample['Ib'] + df_sample['Ic']
        df_sample['V0'] = df_sample['Va'] + df_sample['Vb'] + df_sample['Vc']
        
        # Align features to expected training order
        sample_input = df_sample[expected_features]
        
        # Run inference
        single_pred = model.predict(sample_input)[0]
        
        # Map predictions to targets [G, C, B, A]
        fault_status = dict(zip(target_names, map(int, single_pred)))
        
        return jsonify({
            'status': 'success',
            'inference_result': fault_status
        })
        
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True, port=5000)