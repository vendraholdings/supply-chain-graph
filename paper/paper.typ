#set document(title: "Mapping the Semiconductor Supply Chain from Public Filings")
#set page(paper: "a4", margin: (x: 2.3cm, y: 2.4cm), numbering: "1")
#set text(font: "New Computer Modern", size: 10pt)
#set par(justify: true, leading: 0.62em)
#set heading(numbering: "1.1")
#show heading.where(level: 1): set text(size: 13pt)
#show heading.where(level: 2): set text(size: 11pt)
#show figure.caption: set text(size: 9pt)
#show figure: set block(breakable: false)
#let fig(n, name, w, cap) = {
  counter(figure.where(kind: image)).update(n - 1)
  [#figure(image("figures/" + name + ".png", width: w), caption: cap, placement: auto)#label("fig" + str(n))]
}

#align(center)[
  #text(size: 17pt, weight: "bold")[Mapping the Semiconductor Supply Chain from Public Filings] \
  #v(2pt)
  #text(size: 12pt)[Network structure, shock propagation and customer-to-supplier return predictability]
]
#v(10pt)

#pad(x: 1.2cm)[
  *Abstract.* This paper builds a directed supplier-to-customer graph of 111 listed semiconductor companies in eleven tiers, from wafers and gases to hyperscalers,
  with the filing, date and source of every edge, and uses it to ask how shocks and returns move across the chain. At the build date the graph holds 64 sourced
  supplier-customer pairs (306 edge-years) covering 42 companies: 16 pairs curated by hand and the rest extracted by sentence rules from
  602 annual reports of 56 SEC filers, without confirmation by a language model. A generalised variance
  decomposition of eleven tier indices gives a total spillover index of 42.6%, with equipment and packaging and substrates as net transmitters. Lead-lag correlations
  between tiers are dominated by the order of market closing times. An event study of 3,940 large one-day price moves finds no abnormal return in
  suppliers, customers or companies two links away that differs from zero at the 5% level; the test has the correct size on random dates. A customer-momentum strategy in the style
  of Cohen and Frazzini earns an annualised net Sharpe ratio of 0.01 (0.15 before costs), which is not distinguishable from strategies built on randomly rewired edges, and the
  probability of backtest overfitting over the declared grid of specifications is 0.78. The data give no evidence that the disclosed links predict returns.
]

= Introduction

Semiconductor production passes through many specialised firms: materials and gases, wafers, deposition, etch and lithography equipment, electronic design automation and intellectual property,
foundries and memory makers, packaging, chip designers, and finally the system makers and cloud providers that buy the chips. Most of these links are not recorded in any
single public dataset. They are, however, disclosed in annual reports whenever a customer exceeds a materiality threshold, for example 10% of revenue in US filings.
Cohen and Frazzini @cohen2008 showed for US firms that the returns of a supplier are predicted by the lagged returns of its customers, which they attribute to limited investor attention to economic links.
Menzly and Ozbas @menzly2010 document related cross-predictability between industries linked through the input-output structure.

This project asks three questions with only free data. First, can a graph of listed semiconductor companies be built with a documented source and date on every edge? Second, how large are
the spillovers between tiers and which tiers transmit and receive them? Third, do the sourced links predict returns, either through customer momentum or through abnormal returns of linked companies after large price moves?
Negative and uninformative results are reported as such. The exported graph is intended for reuse by later projects that map news items to affected companies.

The edges used here are extracted by rules alone; the language-model confirmation step is implemented but was not run, and the effect of that on precision is discussed in Section 5.

= Data and graph construction

== Universe and prices

A seed list of 114 tickers was verified against a price vendor: each ticker had to exist, be an equity, match its name, and have at least three years of history and a recent last price.
3 were dropped, each with a recorded reason (two have no price history and one has vendor history ending on 2 October 2026), leaving 111 listings, one per company on its home exchange, in the
eleven tiers of @fig2. Three listings with short histories are kept and flagged. The universe is defined from today's constituents and is therefore subject to survivorship bias.

Daily adjusted closes from 2015 to 2026-10-30 are converted to US dollars and placed on a common grid of 2,973 US trading days. A listing's return for grid day $t$ is the compounded return over its own
sessions that close after the US close of day $t-1$ and no later than the US close of day $t$. Asian and European sessions on calendar date $t$ therefore fall in grid day $t$ and are known before the US open.
Exchange holidays give missing values rather than zeros. Raw vendor prices are not redistributed with this repository; only derived results are.

== Edges

An edge runs from a supplier to a customer and is stored with the fiscal year, the share of supplier revenue where disclosed, the filing date, the date from which it is usable
(the next trading day after the filing), the date it expires, and the URL of the source document. Only edges with a findable public source are kept; 306 edge-years covering 64 pairs were curated by reading
annual reports and press releases, and 40 of the pairs state a revenue share. Candidates that could not be sourced were dropped, with reasons recorded. An edge is usable from its available date until the next filing that no longer lists it,
or for one year. This point-in-time rule is enforced by tests, including one that delays all filings and checks that the information set never grows.

A second source extracts customer-concentration statements automatically from the text of 10-K and 20-F filings. Sentence rules find a named customer in the universe followed by a share of revenue
(or a statement that the customer exceeded a disclosure threshold). Shares stated for several customers "respectively" are paired by position; aggregate shares ("top ten customers"), market shares and matches that cross a sentence boundary are rejected.
The client reads the EDGAR submissions index of each SEC filer in the universe and every 10-K and 20-F filed since 2015 (602 reports of 56 companies) at no more than five requests per second, which gave 974 candidate passages and 276 edge-years
(59 pairs; 9 of these repeat a curated edge-year, for which the curated row is kept). A second pass by a locally run open-weights language model, which must agree with the rules before an edge is accepted, is implemented and tested on cached responses, but no model endpoint was available for this build;
the automated edges are therefore marked as not model-confirmed with medium confidence. On ten real filings read in full, the rules before the last revision recovered 5 of about 15 named pairs, with no false positive; the misses are dollar tables.
A read of 30 randomly drawn edge-years against their source sentences found no clear misattribution after the revision, but this is a check of the extracted sentence and not of the filing page, and a first draw before the revision had found four
wrong attributions in 24, which the revision removes. Table-like layouts and customers named only in a table are missed, so the extracted set is incomplete.

== Graph

The graph combines the curated and automated edges: 42 connected companies and 64 edges, weighted by disclosed revenue share where available and by the tier median otherwise.
The first month in which any edge is usable is 2015-02, and at most 19 suppliers have a signal in any month.

= Methods

*Centrality.* PageRank with damping 0.85, weighted in-strength and out-strength, and betweenness, computed on the directed graph @page1999. Rank correlations with market capitalisation use Spearman's coefficient.

*Customer momentum.* Following Cohen and Frazzini, each month end a supplier's signal is the revenue-weighted return in US dollars of its customers over the month; suppliers are sorted into terciles, the top tercile is bought
and the bottom sold, equal-weighted and held one month. Costs are 10 basis points one way for US, European and Japanese listings, 25 basis points for Taiwan, Korea and Hong Kong, and 50 basis points a year on short positions.
Alpha is estimated with Newey-West standard errors @neweywest1987 and by Fama-MacBeth cross-sectional regressions @famamacbeth1973. The primary specification is fixed in the configuration file before any result is computed, and a declared grid of 48 variants
(lag, quantiles, weighting, edge set, holding period) is run so that the multiple-testing diagnostics of the deflated Sharpe ratio @bailey2014 and of the probability of backtest overfitting can be applied.
A falsification test rewires the edges at random within tiers, preserving degrees.

*Event study.* An event is a one-day move of a listing larger than three trailing 60-day standard deviations. For each linked listing, day 0 is its first session closing after the event, so that an after-close US move is day 0 for Asia
on the next calendar date. Abnormal returns are the US-dollar return minus a market model with the local index, a global semiconductor index and a lagged term @dimson1979, estimated on days $-250$ to $-30$. Returns are sign-adjusted, so that a
supplier moving with the event node counts positively whatever the direction of the shock. The test statistic is the standardised cross-sectional test @bmp1991 adjusted for event clustering @kolari2010.
The size of the test is checked on 1,000 draws of random listings and dates with the same number of events as each real set.

*Spillover.* The generalised forecast-error variance decomposition of Diebold and Yilmaz @dy2012, based on generalised impulse responses @pesaran1998, is computed from a vector autoregression of the eleven tier indices (cap-weighted, in US dollars,
residualised on the US market) with the lag chosen by BIC (at most three) and a ten-day horizon. The total spillover index is the share of forecast-error variance that comes from other tiers. Rolling estimates use 250-day windows; weekly data are used as a robustness check.
The estimator is validated on a simulated autoregression with a known decomposition, matching it within 0.02.

*Lead-lag.* Cross-correlations of tier returns at lags from $-5$ to $+5$ grid days for all 110 ordered tier pairs, with 95% intervals from a stationary bootstrap @politis1994.

= Results

== Network structure

@fig2 shows the universe by tier and listing country and the country pairs of the sourced edges. The sourced graph is concentrated in equipment and its subsystem suppliers, memory, and a few chip designers, and is
dominated by US listings. Among the 42 connected companies, PageRank correlates with market capitalisation at Spearman's $rho$ = 0.64 (n = 42), while weighted out-strength correlates negatively, at -0.45, because the
suppliers with the largest out-strength are not the largest companies (@fig3). @fig4 aggregates the disclosed shares by tier pair. These statistics describe the sample of sourced edges, which was selected by disclosure thresholds and by what could be found,
and are not estimates of the structure of the industry.

#fig(2, "fig02_country_tier_light", 88%, [*Universe and sourced pairs by country and tier.* Left: number of listings by tier and listing country. Right: sourced supplier-customer pairs by country of the two listings.])

#fig(3, "fig03_centrality_light", 88%, [*Centrality of the connected companies.* PageRank of the 42 companies with at least one sourced edge, and PageRank rank against market-capitalisation rank. Ties are at average rank.])

#fig(4, "fig04_tier_flow_light", 50%, [*Disclosed revenue share by tier pair.* Sum of disclosed supplier-revenue shares of sourced pairs, supplier tier in rows and customer tier in columns; blank cells have no sourced pair.])

== Customer momentum

The primary specification has 139 active months. Its long-short return has an annualised Sharpe ratio of 0.15 before costs and 0.01 after costs and borrow (-0.13 at twice the cost); the
Newey-West alpha against the market and the semiconductor index is +0.2% a year ($t$ = 0.02), and the Fama-MacBeth slope has $t$ = 1.5 (@fig5). Across 1,000 degree-preserving rewirings of the edges within tiers, the Sharpe ratio has mean -0.03 and
standard deviation 0.21, and a share 0.43 of the draws reach or exceed the observed value, so the strategy is not distinguishable from one built on random links (@fig6, left).
Of the declared grid of 48 specifications, 24 use the automated edges and are observed in every month; the others rely on the sparse curated set and are excluded from the overfitting diagnostics. For those 24, the deflated Sharpe ratio of the best specification is 0.09
(the expected maximum Sharpe ratio of the grid under no skill is 0.41 a year, @fig6 right), the probability of backtest overfitting is 0.78, and the share of splits in which the selected specification loses money out of sample is 93%.

#fig(5, "fig05_momentum_light", 58%, [*Customer-momentum strategy.* Growth of one unit in the primary specification (terciles, equal weight, curated and automated edges, one-month holding), gross and net of costs and borrow.])

#fig(6, "fig06_momentum_placebo_light", 92%, [*Falsification and multiple testing.* Left: net Sharpe ratio of 1,000 rewired-edge strategies and the observed value. Right: net Sharpe ratio of each fully observed specification and the expected maximum under no skill.])

== Event study

Of the 3,940 price shocks, 850 events have a direct supplier in the graph, 947 a direct customer, and 842 a company two links away. The cumulative average abnormal return over days 0 to 20 is
+0.6% for suppliers (adjusted $t$ = 1.1), +0.6% for customers ($t$ = 1.8) and +0.4% for companies two links away ($t$ = 1.5); none is significant at the 5% level (@fig7); all three are positive and below one per cent.
The event nodes themselves show a sign-adjusted CAAR of +7.1%, which is mechanical because events are selected on large moves.
On random dates the 5% test rejects in 5.6%, 5.6% and 4.5% of draws for the three groups, which indicates that the test has the correct size (@fig8). The events within a group overlap in time and share common shocks, which the adjustment only partly addresses, and a mean effect below one per cent would need many more independent events to detect, so the absence of significance is not evidence of absence.

#fig(7, "fig07_event_fans_light", 92%, [*Abnormal returns of linked companies after price shocks.* Sign-adjusted cumulative average abnormal return over days 0 to 20, by relationship to the company with the shock, with tiers as thin lines.])

#fig(8, "fig08_placebo_light", 100%, [*Placebo distribution on random dates.* Cumulative average abnormal return at day 20 for random listings and dates, with the observed value marked.])

== Spillovers between tiers

The static total spillover index of the eleven tier indices is 42.6% (49.4% with weekly data), so slightly more than two fifths of the ten-day forecast-error variance of a tier comes from other tiers. The rolling index ranges between 30.4% and 55.0%
across 545 windows, and the VAR is stable in every window (@fig10). The largest net transmitters are equipment (T05, +11.6 points) and packaging and substrates (T09, +5.5); the largest net receivers are foundry (T07, -7.5) and industrial gases (T02, -4.6) (@fig9).
Tier indices use current market capitalisation as static weights, which favours today's large companies.

#fig(9, "fig09_spillover_table_light", 70%, [*Generalised forecast-error variance decomposition, ten-day horizon.* Each row sums to 100% over the originating tiers; the colour scale is capped at 20% so that off-diagonal cells are legible.])

#fig(10, "fig10_spillover_rolling_light", 85%, [*Rolling total spillover index and net directional spillover by tier.* 250-day windows of residualised daily returns on the US trading-day grid.])

== Lead-lag relations

Of the non-zero-lag intervals, 12.7% exclude zero, against 5% expected under the null of no relation (@fig11). The strongest relation is T10 leading T07 at one day (correlation 0.40). Because the grid places the Asian
and European sessions of a calendar date before the US open, a lag of one day between a mainly Asian tier and a mainly US tier largely reflects closing-time order. The median same-day correlation between the Asian and US halves of a tier is 0.12, reported separately.
The lead-lag results are therefore not evidence of information flow along the supply chain.

#fig(11, "fig11_leadlag_light", 82%, [*Lead-lag cross-correlations between tiers.* Row tier leads column tier at positive lags; intervals are 95% stationary-bootstrap bands.])

= Discussion and limitations

The price-based results describe how the tiers move together and do not depend on the edge set. They show substantial common movement (a total spillover index of 42.6%) with equipment and packaging as net transmitters. The edge-based results are negative:
the customer-momentum strategy does not beat random links after costs, the best of the declared specifications does not survive deflation, and the linked-company event study finds no significant abnormal return. These are results about this edge set, which is small and
partly unconfirmed, and about this period and universe; they do not show that economic links carry no information elsewhere.

Limitations:
- *Edge coverage.* Edges come from disclosure thresholds, which favour large customers and suppliers with concentrated sales. Customers reported anonymously cannot be resolved without curation. Coverage of Japan, Korea, Taiwan and Hong Kong filers relies on manual curation.
- *Edge precision and recall.* The automated edges are not model-confirmed and the rules miss table-only disclosures, so both false and missing links are possible. A strategy built on noisy links is biased towards zero.
- *Short sample.* The momentum strategy has 139 monthly observations, so a Sharpe ratio has a standard error near 0.3.
- *Survivorship.* The universe and the tier weights use today's constituents.
- *Timing.* Filing dates are the dates of the documents; some sources are mirrors of the original filing, and one filing date is an auditor report date. Edges become usable the next trading day.
- *Costs and constraints.* Costs are stylised; short-sale constraints and borrow availability in Asian markets are not modelled beyond a flat fee.
- *Mechanical lead-lag.* Time-zone order explains the largest lead-lag correlations.
- *Single data vendor.* Prices come from one free vendor and were not cross-checked against a second source.

#bibliography("refs.bib", style: "apa", title: "References")
