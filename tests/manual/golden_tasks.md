# Golden Tasks

Use these tasks for repeated human review rounds.

## Text Queries

1. Query: `family suv cargo monthly payment`
   - Expected good cars: practical SUV entries with AWD/family/cargo language
   - Verify: top 3 contains an obvious family SUV and finance notes are useful

2. Query: `electric sedan range charging`
   - Expected good cars: EV sedans
   - Verify: top result or top 3 contains a clearly electric sedan and range/charging info appears

3. Query: `truck towing payload business financing`
   - Expected good cars: work trucks or towing-focused pickups
   - Verify: top result or top 3 contains a truck and the finance snippet does not feel consumer-car-specific

4. Query: `luxury electric sedan`
   - Expected good cars: higher-end EV sedans
   - Verify: result image matches the category and not an SUV/truck

5. Query: `cheap awd family car`
   - Expected good cars: AWD SUV/crossover or AWD family-friendly cars
   - Verify: top results are not dominated by trucks or EV sedans

## Image Queries

6. Input: side/front image of an SUV
   - Expected good cars: visually similar SUVs
   - Verify: result image similarity feels plausible

7. Input: sedan reference image
   - Expected good cars: sedan entries
   - Verify: no truck dominates the first result

8. Input: pickup reference image
   - Expected good cars: truck entries
   - Verify: towing/payload-oriented vehicles rise to the top

## PDF Queries

9. Input: brochure PDF for a sedan
   - Expected good cars: sedan entries
   - Verify: PDF upload works and results are coherent

10. Input: brochure PDF for a truck
    - Expected good cars: truck entries
    - Verify: spec-heavy truck language influences results

## Finance File Queries

11. Input: finance markdown emphasizing low payment and APR
    - Expected good cars: entries with finance-friendly positioning
    - Verify: finance snippets appear relevant

12. Input: finance markdown emphasizing commercial use
    - Expected good cars: truck or work-oriented vehicles
    - Verify: the top result is not an unrelated sedan

## UI Checks During Every Task

13. Verify result cards show car images
14. Verify expanding details shows summary, specs, and finance
15. Verify there are no obvious duplicate junk results in the first page of output
