"""EWICRS film narration: every line is cued to a visual beat inside its scene.

Scene durations (seconds) and global start offsets are defined here so that the
video, the narration track and the music all share one timeline.

Pronunciation hints used for the offline TTS engine (display text stays normal):
  EWICRS -> "E. W. I. C. R. S."   ESP32 -> "E.S.P. 32"   pH -> "P.H."
"""

FPS = 30
SCENES = [  # key, title, duration (s)
    ("s1", "The Water Intelligence Problem", 45.0),
    ("s2", "Four-Layer Intelligence", 30.0),
    ("s3", "Physical Digital Twin", 40.0),
    ("s4", "Experiment 1 - Model Failure and Revision", 40.0),
    ("s5", "Experiment 2 - Pollution Detection", 40.0),
    ("s6", "Experiment 3 - Climate Resilience", 40.0),
    ("s7", "Physical Implementation Pathway", 40.0),
    ("s8", "Conclusion", 14.9),
]
DUR = {k: d for k, _, d in SCENES}
OFFSET = {}
_t = 0.0
for k, _, d in SCENES:
    OFFSET[k] = _t
    _t += d
TOTAL = _t  # 289.9 s

# (scene, cue time inside scene, spoken text, display text if different)
NARRATION = [
    ("s1", 6.6, "Water sustains every community. Yet one in four people, 2.1 billion, still lack safely managed drinking water."),
    ("s1", 15.4, "Reservoirs rise and fall with rain. Drought and storms push infrastructure past its limits."),
    ("s1", 22.0, "And pollution can enter a network, and travel through it before anyone notices."),
    ("s1", 28.2, "Monitoring and prediction are not enough. When reality drifts away from the model, who notices? That is the intelligence gap."),
    ("s1", 38.0, "This is E. W. I. C. R. S."),

    ("s2", 0.3, "Four layers of intelligence."),
    ("s2", 3.6, "Epistemic intelligence tracks what the system knows, and how reliable that knowledge is."),
    ("s2", 10.2, "Scientific intelligence tests models against physical laws."),
    ("s2", 15.0, "Predictive intelligence forecasts storage, demand, climate and pollution risk."),
    ("s2", 20.0, "Adaptive intelligence simulates alternatives, and recommends approved actions."),
    ("s2", 25.4, "Feedback then closes the loop."),

    ("s3", 0.6, "The digital twin is a time-stepped simulation of a school water system: catchment, tanks, pipes, demand and sensors."),
    ("s3", 9.4, "Storage changes with inflow and rainfall, minus demand and leakage."),
    ("s3", 14.6, "It also models pollutant entry, water-quality sensors, and treatment and isolation points."),
    ("s3", 22.0, "All of this data is synthetic."),
    ("s3", 25.0, "Here is the dashboard from this run."),
    ("s3", 28.4, "Tank storage."),
    ("s3", 30.3, "Rainfall."),
    ("s3", 32.2, "Temperature."),
    ("s3", 34.1, "Water quality."),
    ("s3", 35.9, "Actual versus predicted."),
    ("s3", 38.1, "Risk score."),

    ("s4", 0.5, "Experiment one: does justified model revision improve later prediction?"),
    ("s4", 5.6, "In the simulated physical system, an extra water loss is hidden from the inference model."),
    ("s4", 12.2, "The residual is observed minus predicted."),
    ("s4", 15.6, "Here it departs from zero near step 43, peaks near the leakage marker, then reverses sign."),
    ("s4", 22.5, "But a residual alone does not prove the model wrong. The system proposes competing explanations: demand increase, sensor error, unexpected loss, or inflow error."),
    ("s4", 33.4, "A revision is kept only if evidence justifies it, and held-out data agree."),

    ("s5", 0.5, "Experiment two: a simulated pollutant enters the network."),
    ("s5", 5.2, "The plume travels through the pipes with transport delay, reaching upstream and downstream sensors at different times."),
    ("s5", 12.9, "Five explanations compete: upstream contamination, local contamination, runoff, sensor failure, or unresolved cause."),
    ("s5", 21.4, "Each predicts a different pattern across sensors. The system weighs the evidence, and can conclude that evidence is insufficient to distinguish hypotheses."),
    ("s5", 31.9, "A simulated intervention then isolates the affected zone, and quality is verified afterwards."),

    ("s6", 0.6, "Experiment three tests climate resilience, starting with normal rainfall."),
    ("s6", 6.2, "Then, prolonged drought."),
    ("s6", 11.2, "Extreme rainfall."),
    ("s6", 16.2, "Increased demand."),
    ("s6", 21.2, "And drought combined with increased demand."),
    ("s6", 26.4, "Three strategies face identical conditions: fixed rules, forecasting, and epistemic adaptation."),
    ("s6", 34.0, "Compared on reliability, unmet demand, shortage duration and recovery."),

    ("s7", 0.6, "The physical pathway is a safe, isolated tabletop prototype, not connected to any school water system."),
    ("s7", 8.0, "An E.S.P. 32 reads level, turbidity, P.H. and temperature sensors, and drives a low-voltage pump between two containers."),
    ("s7", 19.0, "Readings are calibrated against reference standards, then flow to a Python backend, the digital twin, and the intelligence engine."),
    ("s7", 27.8, "Only approved commands return to the pump, with manual override and an emergency stop."),
    ("s7", 34.6, "This hardware pathway is planned, not yet validated."),

    ("s8", 0.8, "The future of intelligent water management is not just predicting what will happen."),
    ("s8", 6.5, "It is understanding how reliable our knowledge is, testing it against reality, and adapting when reality proves us wrong."),
]

# Text shown in narration.txt (readable form of the spoken lines)
READABLE = {
    "E. W. I. C. R. S.": "EWICRS",
    "E.S.P. 32": "ESP32",
    "P.H.": "pH",
}
