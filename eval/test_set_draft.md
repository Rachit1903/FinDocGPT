# FinDocGPT evaluation-set draft (40 items)

Review this set before it becomes `test_set.json`. Each item needs a filing-backed reference answer and source filing/chunk before scoring; this prevents fabricated RAGAS results.

## AAPL

1. **Factual** — What did Apple identify as the main drivers of Services net sales in fiscal 2025?
2. **Factual** — What risk factors did Apple disclose regarding supply-chain concentration?
3. **Comparison** — Compare Apple's Products and Services gross-margin percentages in fiscal 2025.
4. **Computation** — Calculate the change in Apple's total gross-margin percentage from 2024 to 2025.

## AMZN

5. **Factual** — What factors did Amazon cite for the change in North America operating income in the latest annual filing?
6. **Factual** — What did Amazon disclose about AWS revenue growth drivers?
7. **Comparison** — Compare Amazon's North America and International segment operating results.
8. **Computation** — Calculate Amazon's year-over-year AWS net-sales growth using the relevant segment table.

## GOOGL

9. **Factual** — What drove Google Services revenue growth in the latest annual filing?
10. **Factual** — What did Alphabet disclose about AI infrastructure investment?
11. **Comparison** — Compare Google Services and Google Cloud operating income trends.
12. **Computation** — Calculate Google Cloud's operating-margin change between the two latest annual periods.

## META

13. **Factual** — What did Meta attribute to the change in advertising revenue?
14. **Factual** — What did Meta disclose about Reality Labs operating losses?
15. **Comparison** — Compare Meta's Family of Apps and Reality Labs operating performance.
16. **Computation** — Calculate Meta's revenue growth rate between the two latest annual periods.

## NFLX

17. **Factual** — What factors did Netflix cite for revenue growth in the latest annual filing?
18. **Factual** — What did Netflix disclose about content obligations and spending?
19. **Comparison** — Compare Netflix's operating margin across the two latest annual periods.
20. **Computation** — Calculate Netflix's operating-margin percentage-point change year over year.

## MSFT

21. **Factual** — What did Microsoft attribute to Azure revenue growth?
22. **Factual** — What did Microsoft say caused the gross-margin percentage change?
23. **Comparison** — Compare the gross-margin drivers for Microsoft's Productivity and Business Processes versus Intelligent Cloud segments.
24. **Computation** — Calculate the change in Microsoft total gross-margin percentage between the two latest annual periods.

## NVDA

25. **Factual** — What did NVIDIA identify as the drivers of Data Center revenue growth?
26. **Factual** — What did NVIDIA disclose about inventory provisions or purchase obligations?
27. **Comparison** — Compare NVIDIA's Data Center and Gaming revenue trends.
28. **Computation** — Calculate NVIDIA's gross-margin percentage change year over year.

## CRM

29. **Factual** — What did Salesforce attribute to subscription and support revenue growth?
30. **Factual** — What did Salesforce disclose about remaining performance obligations?
31. **Comparison** — Compare Salesforce's subscription-and-support revenue with professional-services revenue trends.
32. **Computation** — Calculate Salesforce's operating-margin change between the two latest annual periods.

## ADBE

33. **Factual** — What did Adobe attribute to Digital Media segment revenue growth?
34. **Factual** — What did Adobe disclose about the impact of generative AI on its products or strategy?
35. **Comparison** — Compare Adobe's Digital Media and Digital Experience segment revenue trends.
36. **Computation** — Calculate Adobe's total revenue growth rate between the two latest annual periods.

## UBER

37. **Factual** — What did Uber identify as drivers of Mobility gross bookings growth?
38. **Factual** — What did Uber disclose about its profitability or operating-income change?
39. **Comparison** — Compare Uber's Mobility and Delivery segment revenue trends.
40. **Computation** — Calculate Uber's year-over-year total revenue growth using the annual filing table.

## Review criteria

- Use the specific filing and fiscal period already ingested; do not mix annual and quarterly figures.
- Reference answers must be factual, concise, and cite source chunks or table IDs.
- Preserve the 14 factual, 13 comparison, and 13 computation category balance.
