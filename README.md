# Deep-Residual-Learning-for-Image-Recognition-Implementation
Based on paper by Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun - https://arxiv.org/pdf/1512.03385.pdf

Insights:
1. The residual has been defined as I(x) = H(x) - F(x)

where I(x) is identity mapping which is I(x) =x,H(x) the desired mapping ,F(x) the mapping which network layers where able to achieve according to their inputs.

H(x) = F(x) +I(x) is our objective and only F(x) is changing everytime assuming H(x) and I(x) as constants.

Rather than learning F(x) in a simple deep network which can have degradation problem its helpful for the model to give accurate results. I(x) is adding the deficiencies in F(x) at each skip connection if it is less than H(x) and absorbing all the errors in F(x) is more than H(x).

2. One of the basic principles involved is the idea to build a deeper network by simply repeating this module i.e the smaller network.

3.
