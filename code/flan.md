# Flan

The Flan model is described here: https://huggingface.co/google/flan-t5-small

In order to explore the LLM without having to engage in the difficult process of fine-tuning, we explored the effect of altering the prompt on the model accuracy.

Here is a table of sample results using the prompt: "Is the following sentence positive or negative? "

```
                 precision    recall  f1-score   support

Negative Review       0.83      0.85      0.84       533
Positive Review       0.85      0.83      0.84       533

       accuracy                           0.84      1066
      macro avg       0.84      0.84      0.84      1066
   weighted avg       0.84      0.84      0.84      1066
```
