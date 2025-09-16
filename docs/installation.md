[HOME](../README.md)

# Installation Guide

## Step 1

First check if your data comes with precomputed distances and if you are going to want to use the default [driving connections](./graph.md) when building your graph.
Depending on your choices you will need to install the library with torch. To see what your use case requires check the table below and copy the command.

| data has distances | use driving edges | installation mode                   |
|--------------------|-------------------|-------------------------------------|
|       YES          |        YES        |`pip install .[torch]`|
|       YES          |        NO         |`pip install .[torch]`|
|       NO           |        YES        |`pip install .[torch]`|
|       NO           |        NO         |   `pip install .`    |

## Step 2

> RUN your command from the root of this project

## Step 3

The library should now be installed to check you can test with this:

```text
python -c "import multimodalrouter; print(dir(multimodalrouter))"
```

The result should look similar to this

```python
[
'EdgeMetadata', 
'Hub', 
'OptimizationMetric', 
'Route', 
'RouteGraph', 
'VerboseRoute', 
'...', 
'preprocessor', 
'utils'
]
```

