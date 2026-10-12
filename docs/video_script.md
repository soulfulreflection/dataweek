# 7-minute video script

Upload to YouTube as *Unlisted*, check the link in a private browser window, and submit the link in a Notepad file
(not the .mp4).

| Time | Speaker | Screen | Script |
|---|---|---|---|
| 0:00-0:40 | A | Title slide | Introduce the team. "Reservoir engineers decide multi-billion-dollar investments on forecasts that come from history-matched models. History matching is slow because every guess needs a simulator run. We built an ML workflow that does it in minutes and forecasts the next 20 years." |
| 0:40-1:30 | A | Slides 3-4 | Background: reservoir model, porosity and permeability, the 4 uncertainty parameters, 100 simulated cases, 10 years of observed data. The four objectives. |
| 1:30-2:30 | B | Slides 5-6 | Data insights: wells on liquid-rate control; gas is a constant 0.363 × oil; porosity dominates. Show the water-cut collapse: "low-porosity cases show us the future of the field". |
| 2:30-3:30 | B | Slides 7-8 | Proxy: PCA compresses each production curve into 12 numbers; Gaussian processes predict them from the 4 parameters. Beats an MLP, RBF and polynomial models; R² 0.997 on unseen cases; it also knows its own uncertainty. |
| 3:30-4:30 | C | Slide 9 | History match: 131k candidate reservoirs screened in seconds, then MCMC. Porosity is pinned to 1.349; fault transmissibility can't be identified from production data, which is why we deliver 10 cases. Blind test: we recover known parameters of test cases. |
| 4:30-5:30 | C | Slides 10-11 | Forecast: the simulations stop in 2008, so we backtested 5 forecasting methods on 100 cases; the oil-cut power law wins (1.2% error). Key insight: the real field waters out faster than any model realisation, so we anchor on the observed trend. Trusting the model blindly would overstate remaining oil by 29%. |
| 5:30-6:20 | D | Power BI | Dashboard walkthrough: KPIs (55.9 MMSTB by 2028, water cut 90%), forecast page, history-match page, model-quality page, business page. |
| 6:20-7:00 | D | Slide 12 | Conclusions, next steps (wider prior, simulator check of the 10 cases, real economics). How the team collaborated (Git, task split). Thank you. |

Recording tips: 1080p and a quiet room; rehearse once with a timer; keep each speaker under 90 seconds; zoom
into the charts.
