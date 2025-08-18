# IntramarketDifference

First thing: make sure I understand how it works.

Each year:
1. Load all available coins
2. Simulate pair tradings with each combinaton
3. Select only those with good profit factor
4. Trade with them the following year
5. When the year of actual trading is finished, repeat the #1 with loading ALL coins (including thiose you didn't use in the trading)

Tips:
- Use .csv with results from each step.
- Check Average profits for the year from longs / short. If they are + -> keep them as it is. If they are - -> invert logic. So, when we get a long trade signal -> make short. Possibly, it will generate income.

XRP - BTC -> don't work together.
All before 2023