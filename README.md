# Intel Submission

Changes to be implemented:
 - Train using conjugate gradient with whole batch size
 - Make a decision regarding the stationary patches (remove or not)
 - Logging the errors
 - Reconstruct a video
 - Try different patch sizes (20x20 etc.) 
 - Analyse why Adagrad, Adam are not working. (Monitor Validation error)

Changes - Done:

- Have all the frames always in the memory, make the whole patch making mechanism internal.
- How to deal with batches, Design a mechanism
- Try with 10 images first.
- Check the difference between patches while comparing 1:1 
