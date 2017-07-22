# ML_CHLNG_3
Machine Learning Challenge 3 from HackerEarth

https://www.hackerearth.com/challenge/competitive/machine-learning-challenge-3/

<h3>Problem Statement</h3>
<p>A leading affiliate network company from Europe wants to leverage machine learning to improve (optimise) their conversion rates and eventually their topline. Their network is spread across multiple countries in europe such as Portugal, Germany, France, Austria, Switzerland etc.</p>
<p>Affiliate network is a form of online marketing channel where an intermediary promotes products / services and earns commission based on conversions (click or sign up). The benefit companies sees in using such affiliate channels is that, they are able to reach to audience which doesn’t exist in their marketing reach. </p>
<p>The company wants to improve their CPC (cost per click) performance. A future insight about an ad performance will give them enough headstart to make changes (if necessary) in their upcoming CPC campaigns. </p>
<p>In this challenge, you have to predict the probability whether an ad will get clicked or not. </p>
<h3><a href="https://he-s3.s3.amazonaws.com/media/hackathon/machine-learning-challenge-3/predict-ad-clicks/205e1808-6-dataset.zip">Download Dataset</a></h3>
<h3>Data Description</h3>
<p>You are given three files to download: train.csv, test.csv and sample_submission.csv  Variables in this data set are anonymized due to privacy. <br />
The training data is given for 10 days ( 10 Jan 2017  to  20 Jan 2017). The test data is given for next 3 days. </p>
<table class="pd-table">
<thead>
<tr>
<th>Variable</th>
<th>Description</th>
</tr>
</thead>
<tbody>
<tr>
<td>ID</td>
<td>Unique ID</td>
</tr>
<tr>
<td>datetime</td>
<td>timestamp</td>
</tr>
<tr>
<td>siteid</td>
<td>website id</td>
</tr>
<tr>
<td>offerid</td>
<td>offer id  (commission based offers)</td>
</tr>
<tr>
<td>category</td>
<td>offer category</td>
</tr>
<tr>
<td>merchant</td>
<td>seller ID</td>
</tr>
<tr>
<td>countrycode</td>
<td>country where affiliates reach is present</td>
</tr>
<tr>
<td>browserid</td>
<td>browser used</td>
</tr>
<tr>
<td>devid</td>
<td>device used</td>
</tr>
<tr>
<td>click</td>
<td>target variable</td>
</tr>
</tbody>
</table>
<p><br /></p>
<h3>Submission</h3>
<p>A participant has to submit a zip file containing your ID and predicted probabilities in a csv format. Check the sample submission file for format.</p>
<pre class="prettyprint"><code>ID,click
IDE4beP,0.125
IDfo26Y,0.015
IDYZM6I,0.201
ID8CVw1,0.894
IDPltMK,0.157

</code></pre>
<p><br /></p>



