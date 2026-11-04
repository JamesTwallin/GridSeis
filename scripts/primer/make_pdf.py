"""Assemble the grid-frequency primer PDF from the figures made by main.py."""
import os
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_LEFT
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, PageBreak, Table, TableStyle, KeepTogether
from PIL import Image as PILImage

HERE = os.path.dirname(os.path.abspath(__file__)); os.chdir(HERE)
INK = colors.HexColor('#1f2937'); MUTED = colors.HexColor('#6b7280')
ss = getSampleStyleSheet()
h1 = ParagraphStyle('h1', parent=ss['Title'], fontName='Helvetica-Bold', fontSize=22, leading=26, textColor=INK, alignment=TA_LEFT, spaceAfter=4)
sub = ParagraphStyle('sub', parent=ss['Normal'], fontName='Helvetica', fontSize=10.5, leading=14, textColor=MUTED, spaceAfter=10)
h2 = ParagraphStyle('h2', parent=ss['Heading2'], fontName='Helvetica-Bold', fontSize=13.5, leading=17, textColor=INK, spaceBefore=10, spaceAfter=4)
body = ParagraphStyle('body', parent=ss['Normal'], fontName='Helvetica', fontSize=10.5, leading=14.5, textColor=INK, spaceAfter=7)
cap = ParagraphStyle('cap', parent=body, fontSize=9, leading=12, textColor=MUTED, spaceBefore=2, spaceAfter=10)
cell = ParagraphStyle('cell', parent=body, fontSize=9.5, leading=12, spaceAfter=0)
doc = SimpleDocTemplate('grid_frequency_primer.pdf', pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm, topMargin=16 * mm, bottomMargin=16 * mm,
                        title='How grid frequency behaves as the generation mix changes', author='James Twallin')


def fig(path, width_mm, caption):
    w, h = PILImage.open(path).size; wd = width_mm * mm
    return [Image(path, width=wd, height=wd * h / w), Paragraph(caption, cap)]


def P(t, s=body): return Paragraph(t, s)


def table(rows, colw):
    t = Table([[Paragraph(c, cell) for c in r] for r in rows], colWidths=colw, repeatRows=1)
    t.setStyle(TableStyle([('LINEBELOW', (0, 0), (-1, 0), 0.8, INK), ('LINEBELOW', (0, 1), (-1, -1), 0.25, colors.HexColor('#d1d5db')),
                           ('VALIGN', (0, 0), (-1, -1), 'TOP'), ('TOPPADDING', (0, 0), (-1, -1), 3), ('BOTTOMPADDING', (0, 0), (-1, -1), 3)]))
    return [t, Spacer(1, 8)]


S = []
S += [P('How grid frequency behaves as the generation mix changes', h1),
      P('A primer using NESO\'s public one-second frequency data, 2021 to 2026. James Twallin, 30 October 2026.', sub)]
S += [P('1. What the frequency is', h2),
      P("Every generator on the British grid spins in step at 50 cycles a second. The frequency you can measure at any socket is a single "
        "number for the whole island, and it moves whenever generation and demand are not exactly equal. More demand than generation "
        "and it falls; more generation and it rises. It is the grid's balance gauge, updated every cycle, and it is published by NESO "
        "once a second."),
      P("Three things set how it moves. <b>Inertia</b> is the kinetic energy stored in the spinning machines that are connected: large "
        "turbines in gas, nuclear, coal, biomass and hydro stations. When the balance is upset, that stored energy is the first thing "
        "the grid draws on, and the more of it there is, the more slowly the frequency moves. <b>Response</b> is the deliberate "
        "correction that follows: governors on generators, batteries, and loads that are paid to react within seconds. <b>The "
        "schedule</b> is the half-hourly plan the market runs on; generators ramp between half hours and the frequency shows it."),
      P("Wind, solar, batteries and the undersea cables to other countries connect through power electronics. They can be told to "
        "respond very quickly, but they do not store rotating energy, so they contribute nothing to inertia unless specifically "
        "designed to imitate it.")]
S += fig('fig1_two_days.png', 174,
         "Figure 1. Two full days of one-second frequency on the same scale. The grey band is the normal operating range of plus or "
         "minus 200 mHz. Top: a heavy winter day with high inertia. Bottom: a light summer day with low inertia and a high share of "
         "wind and solar. The vertical lines are half hours.")
S += [PageBreak(), P('2. The mix has changed', h2),
      P("In 2009 almost all of Britain's electricity came from spinning machines. In 2026 less than half does. The rest arrives through "
        "electronics: wind, solar, imports and batteries. This is the single biggest change in how the grid behaves physically, and it "
        "shows up directly in the frequency.")]
S += fig('fig2_mix.png', 174, "Figure 2. Annual generation mix from NESO's historic data. The lower panel is the share from spinning machines.")
S += fig('fig3_inertia.png', 174,
         "Figure 3. Left: NESO's reported system inertia for every half hour since April 2022 against the wind and solar share in that "
         "half hour. Right: inertia by month. The median has fallen and the floor has settled near 110 GVA·s, which is the minimum NESO "
         "chooses to hold by keeping machines running or paying for synchronous condensers.")
S += [PageBreak(), P('3. What that does to the frequency', h2),
      P("The intuitive expectation is that a lighter grid wanders more. The data say something more interesting. Over minutes, the "
        "frequency is actually held slightly tighter when inertia is low, because NESO buys more fast response on those days and "
        "batteries deliver it. Over seconds, the picture reverses: the second-to-second jitter "
        "grows as inertia falls. Inertia is what used to smooth those out, and nothing has fully replaced it.")]
S += fig('fig4_behaviour.png', 174,
         "Figure 4. Two statistics for every half hour from 2022 to 2026, grouped by NESO's reported inertia. Inertia increases to the "
         "left, so the light grid is on the right of each panel. Left: the spread of frequency over the half hour. Right: the typical "
         "change from one second to the next.")
S += fig('fig5_year_sawtooth.png', 174,
         "Figure 5. Left: the spread of frequency within each half hour, by year, with the annual wind and solar share. Right: the average "
         "shape of the frequency within a half hour during 2025, split by the wind and solar share. The dip after the boundary and the "
         "climb towards the next one are generators ramping to their next scheduled output.")
S += [PageBreak(), P('4. Listening to the grid', h2),
      P("Breaking the frequency signal into its component wobbles, from slow swings over minutes to fast ones over seconds, makes the "
        "change in character plain. The light grid is quieter at slow timescales and louder at every timescale faster than about twenty "
        "seconds. It also shows a forest of sharp spikes that the heavy grid barely shows. Those spikes sit at round-number periods: "
        "60, 30, 15, 12, 10, 8, 6 and 5 seconds. Round-number periods point to clocks, not physics. Each spike is a device, or "
        "a fleet of identical devices, whose control loop cycles on a fixed clock, and on a light grid their pushes are large enough to "
        "show in the national frequency. The likely sources are trading and metering intervals and the automated controllers that act on them, all "
        "stepping at the same moment. Dithering, a small random offset in each device's timing, would spread those steps out so they "
        "cancel rather than add."),
      P("The tallest spike, at 3.88 seconds, is a single machine of about 5 MW that has been oscillating since 2022 and switches off "
        "when its site is at work. It is documented separately. It is a useful illustration of the general point: a grid made of many "
        "small electronic units shows its faults as spikes at fixed periods, and those spikes are visible in free "
        "public data.")]
S += [P("The tool for all of this is the Fourier transform, and the idea behind it is simple. Any wiggly line, however messy, can be built by "
        "adding up smooth waves of different speeds. The Fourier transform runs that backwards: give it half an hour of frequency readings and "
        "it tells you how big the wave at each speed is. Draw those sizes against speed and you have a spectrum. A rough slope means the grid is "
        "wobbling at every speed. A sharp spike means something is pushing at one fixed rhythm, and the spike's height is the size of the push. "
        "Figure 6 shows one half hour taken apart this way.")]
S += fig('../../plots/primer/waves.png', 174,
         "Figure 6. Left: five minutes of one half hour of frequency, and three of the waves found inside it: the slow swings, the ringing at "
         "4 to 10 second periods, and the 3.88-second hum. Right: the size of the wave at every speed for that half hour, which is its spectrum.")
S += fig('fig6_spectrum.png', 160,
         "Figure 7. Typical spectrum of the one-second frequency signal for the heaviest and lightest tenth of half hours, 2021 to 2026. "
         "The thin vertical lines mark round-number periods. The shaded band is where the grid's own natural oscillation modes live.")
S += [PageBreak(), P('5. The ringing as a carbon gauge', h2),
      P("Put the pieces together and a useful shortcut appears. Carbon intensity is set by how much gas is burning. Gas turbines are "
        "heavy spinning machines, so a high-carbon half hour is a high-inertia half hour, and a high-inertia grid is quiet at fast "
        "timescales. A low-carbon half hour is a windy one, the grid is light, and it rings. The fast-band ringing of the frequency "
        "signal is therefore an analogue for carbon intensity, and unlike carbon intensity it needs no fuel data: it can be measured "
        "from the mains at any socket in the country, with a microcontroller and a few lines of code."),
      P("Figure 8 shows how good the analogue is. Sorting every half hour since 2021 into twenty groups by ringing, median carbon "
        "intensity falls steadily from about 230 to about 70 gCO<sub>2</sub>/kWh across the groups, and the weekly medians track each "
        "other through five years. The band used, 4 to 10 second periods, sits below the 3.9-second hum and the 2.5 to 3-second winter "
        "bursts, which are machines rather than the grid itself. The relationship was tightest in 2023 and 2024 and loosened in 2025 and 2026, which is itself "
        "informative: as NESO adds synchronous condensers and grid-forming contracts to hold inertia on windy days, the grid rings less "
        "for the same fuel mix. The gauge measures the physics, not the fuel, and the physics is being managed.")]
S += fig('fig7_ringing_vs_carbon.png', 174,
         "Figure 8. Left: carbon intensity against the fast-band ringing of the frequency signal (average spectral amplitude between 0.1 "
         "and 0.25 Hz), every half hour from July 2021 to September 2026 in twenty equal groups. Right: "
         "weekly medians of the two series.")
S += [KeepTogether([P('6. What to take from it', h2)] + table([['Observation', 'What it means'],
            ['Generation from gas, coal, nuclear, biomass and hydro fell from 96% of the total in 2009 to 45% in 2026', "NESO's published inertia fell with it: a typical half hour had 233 GVA·s in 2022 and 165 in 2025."],
            ['40% of half hours in 2025 were below 150 GVA·s, against 10% in 2022; the lowest sit near 115', 'Low-inertia half hours are now routine rather than exceptional.'],
            ['Slow drift has not grown with low inertia', 'Fast response, mostly from batteries, is doing its job over minutes.'],
            ['Second-to-second jitter has grown', 'The first seconds after any disturbance are rougher than they were, and that is where large trips are dangerous.'],
            ['The spectrum shows spikes at round-number periods', 'Clocked controllers are now visible in the national frequency. Monitoring the spectrum rather than the frequency alone shows them early.'],
            ['Fast-band ringing tracks carbon intensity', 'The frequency signal at any socket carries a live estimate of how clean the grid is. That is the idea behind GridSeis.']],
           [70 * mm, 104 * mm]) + [P("Data: NESO System Frequency (one-second readings), NESO Historic Generation Mix, NESO System Inertia outturn. All figures are "
        "produced by main.py in this folder; the half-hourly statistics by build_stats.py.", cap)])]
doc.build(S)
print('written grid_frequency_primer.pdf')
