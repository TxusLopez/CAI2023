# Concept Drift and Uncertainty in Industrial AI

AI-based Digital Twins are at the leading edge of the Industry 4.0 revolution enabled through the Internet of Things and real-time data analytics (streaming). These data are produced in the form of data streams, and should be handled in a continuous fashion. The generation process of such data is usually non-stationary, provoking that data distribution may change, and thus the training of the prediction algorithms become obsolete (phenomenon known as Concept Drift). The early detection of the change (drift) moment is crucial, and the uncertainty estimation is being utilized as drift detector; above all in those scenarios where the true labels are not available. Despite the use of uncertainty estimation for drift detection has already been timidly suggested by a few studies, there is still a lack of solid confirmation in the relation between them. This is the goal of this study.

## Experimentation

You can find the use case in the file "data_gen.py", which generates the sinthetic dataset, processes it in a real-time fashion with the river library, and plot the results.


