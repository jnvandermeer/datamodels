tools and methods to build data models for long-format data


# Updates

There are some things/tricks to be considered using this on data
I had to convert, using error=coerce, any value that was "not measured" - which is one way of coding for missing vlaues
Why write this and not NOT include it?

We have to consider multiple disparate ways of people dealing with data, and somehow keep track of things that happen - and then hope we capture it all
Since plotting '-1', is not a good way - perhaps we shouldremove people that have a nan anywhere, for further consideration

# this is not currently done

- fix NANs
- re-fix things to work on different datasets

- for conda environment, see the conda environment in the datasynthesis module

