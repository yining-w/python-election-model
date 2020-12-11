# Data Skills 2: Homework 4 (ML)
## Voting classification

__Due date: Sunday November 29th before midnight__

In homework 1 we used state-level data on employment in several [NAICS](https://www.naics.com/search/) code sectors from the [Bureau of Economic Analysis](https://apps.bea.gov/iTable/iTable.cfm?reqid=70&step=1&isuri=1).  We have also explored the Pandas DataReader library in order to access the FRED database, among others.  Your goal in this assignment is to use annual state-level data and supervised machine learning to classify states as voting for a Republican or Democrat for president.

1. Use *any* annual state-level data from any two presidential election years (1976 and later) that you like, e.g. 2008 and 2012, 2016 and 2020, etc.  Your data may be retrieved by the program, or downloaded by hand and synched with your repo, or a combination of both.
2. If you have already written functions to use that data, you may reuse those functions here.
3. Merge all data together with the included csv on US presidential election results that I downloaded from [here](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/42MVDX).
4. Your training data will be the first presidential election year, and your testing data will be the second.
5. Use a test harness to assess which model to use (more discussion of this to follow on Wednesday).
6. Fit a supervised ML model (classification) to the data in that way that lets you make predictions, then compare the predictions to the actual outcome.

End your code with a few lines discussing what you found.  Did your model do a good job of predicting the winner?  What accuracy measure did you use to evaluate the model results, and why?  Were you able to reuse any code that you wrote previously, and if so, what worked well or didn't work well?

Do not get too bogged down on actually predicting the winner perfectly.  It's fine to get a mixed result and then discuss how different data or a different model may have improved the results.

As always, don't forget your functions and other good coding practices!
