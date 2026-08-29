# Data Leakage in PPG-based Non-invasive Blood Glucose Prediction

Code for the paper. Three models, five evaluation protocols, three datasets.


| Model                                                                         | Script              | Input                          |
| ----------------------------------------------------------------------------- | ------------------- | ------------------------------ |
| TinyML(ResNet34) ([Zeynali et al., 2025](https://doi.org/10.1038/s41598-024-84265-8)) | `train_resnet.py`   | raw PPG                        |
| CBHFF ([Ali et al., 2025](https://ieeexplore.ieee.org/document/10684156))     | `train_cbhff.py`    | raw PPG                        |
| CatBoost                                                                      | `train_catboost.py` | handcrafted PPG + demographics |


```bash
pip install -r requirements.txt   # Python 3.10+
```

CPU by default. For GPU, set `USE_GPU = True` in `config.py`.

## Data

- **VitalDB** — `python download_vitaldb.py --output-dir dataset/vitaldb --window-seconds 60`
- **MUST** — [Mendeley](https://data.mendeley.com/datasets/37pm7jk7jn/3) into `dataset/MUST/`
- **IPS-PPG** — private; `--dataset IPS`

## Table 5 (within-dataset)

Same order as the paper: dataset → TinyML(ResNet34), CBHFF, CatBoost → RTTS, STTS, PAS, LSPO.

Mean ± std across folds. The paper reports means only due to space. MAE and RMSE are in mmol/L; Accuracy, Zone A/B/A+B, and MARD are in %.

### VitalDB

<table>
<thead>
<tr>
<th>Model</th><th>Split</th><th>MAE</th><th>RMSE</th><th>R2</th><th>Accuracy</th><th>Zone A</th><th>Zone B</th><th>Zone A+B</th><th>MARD</th><th>Pearson's r</th>
</tr>
</thead>
<tbody>
<tr><td rowspan="4">TinyML
  (ResNet34)</td><td>RTTS</td><td>1.4123 ± 0.0188</td><td>1.9224 ± 0.0429</td><td>0.0667 ± 0.0132</td><td>42.75 ± 0.74</td><td>55.13 ± 0.77</td><td>42.29 ± 0.68</td><td>97.43 ± 0.12</td><td>21.76 ± 0.31</td><td>0.2634 (p&lt;0.001)</td></tr>
<tr><td>STTS</td><td>1.4256 ± 0.0338</td><td>1.9498 ± 0.0876</td><td>0.0370 ± 0.0174</td><td>42.26 ± 1.53</td><td>54.79 ± 0.97</td><td>42.64 ± 0.66</td><td>97.43 ± 0.64</td><td>21.87 ± 1.23</td><td>0.1966 (p&lt;0.001)</td></tr>
<tr><td>PAS</td><td>1.5098 ± 0.1017</td><td>2.0914 ± 0.1772</td><td>0.0432 ± 0.0144</td><td>42.03 ± 2.46</td><td>54.93 ± 1.99</td><td>41.82 ± 2.15</td><td>96.74 ± 0.82</td><td>22.56 ± 1.10</td><td>0.2107 (p&lt;0.001)</td></tr>
<tr><td>LSPO</td><td>1.4191 ± 0.0383</td><td>1.9565 ± 0.0859</td><td>0.0303 ± 0.0106</td><td>43.00 ± 2.36</td><td>55.29 ± 1.61</td><td>42.13 ± 1.31</td><td>97.43 ± 0.64</td><td>21.57 ± 1.47</td><td>0.1837 (p&lt;0.001)</td></tr>
<tr><td rowspan="4">CBHFF</td><td>RTTS</td><td>1.4705 ± 0.0109</td><td>1.9899 ± 0.0319</td><td>-0.0001 ± 0.0002</td><td>39.73 ± 0.28</td><td>52.28 ± 0.60</td><td>45.15 ± 0.62</td><td>97.43 ± 0.12</td><td>22.69 ± 0.22</td><td>-0.0016 (p=0.7307)</td></tr>
<tr><td>STTS</td><td>1.4559 ± 0.0506</td><td>1.9807 ± 0.1019</td><td>0.0068 ± 0.0199</td><td>40.77 ± 2.42</td><td>52.72 ± 1.73</td><td>44.70 ± 1.99</td><td>97.43 ± 0.64</td><td>22.33 ± 1.16</td><td>0.0893 (p&lt;0.001)</td></tr>
<tr><td>PAS</td><td>1.5254 ± 0.1096</td><td>2.1106 ± 0.1903</td><td>0.0264 ± 0.0145</td><td>41.82 ± 1.42</td><td>53.63 ± 1.62</td><td>43.11 ± 1.56</td><td>96.74 ± 0.82</td><td>22.74 ± 1.07</td><td>0.1869 (p&lt;0.001)</td></tr>
<tr><td>LSPO</td><td>1.4457 ± 0.0463</td><td>1.9660 ± 0.0847</td><td>0.0209 ± 0.0115</td><td>41.42 ± 1.71</td><td>53.35 ± 1.49</td><td>44.08 ± 1.33</td><td>97.43 ± 0.64</td><td>22.21 ± 1.43</td><td>0.1513 (p&lt;0.001)</td></tr>
<tr><td rowspan="4">CatBoost</td><td>RTTS</td><td>1.2982 ± 0.0109</td><td>1.9062 ± 0.0265</td><td>0.0823 ± 0.0055</td><td>51.21 ± 0.58</td><td>62.59 ± 0.55</td><td>34.83 ± 0.62</td><td>97.43 ± 0.13</td><td>18.58 ± 0.21</td><td>0.3696 (p&lt;0.001)</td></tr>
<tr><td>STTS</td><td>1.3476 ± 0.0480</td><td>1.9518 ± 0.1120</td><td>0.0342 ± 0.0201</td><td>49.12 ± 1.65</td><td>60.65 ± 1.53</td><td>36.78 ± 1.56</td><td>97.43 ± 0.52</td><td>19.34 ± 0.76</td><td>0.2848 (p&lt;0.001)</td></tr>
<tr><td>PAS</td><td>1.4328 ± 0.1067</td><td>2.0908 ± 0.1767</td><td>0.0342 ± 0.0209</td><td>48.37 ± 1.78</td><td>59.73 ± 2.13</td><td>37.00 ± 1.65</td><td>96.73 ± 0.92</td><td>19.86 ± 1.15</td><td>0.3436 (p&lt;0.001)</td></tr>
<tr><td>LSPO</td><td>1.3401 ± 0.0476</td><td>1.9430 ± 0.1109</td><td>0.0428 ± 0.0180</td><td>49.45 ± 1.83</td><td>60.88 ± 1.63</td><td>36.54 ± 1.63</td><td>97.43 ± 0.52</td><td>19.22 ± 0.77</td><td>0.3005 (p&lt;0.001)</td></tr>
</tbody>
</table>

```bash
# TinyML(ResNet34)
python train_resnet.py   --dataset vitaldb --sample-wise --window-seconds 10   # RTTS
python train_resnet.py   --dataset vitaldb --window-seconds 10                # STTS
python train_resnet.py   --dataset vitaldb --pas                              # PAS
python train_resnet.py   --dataset vitaldb                                    # LSPO

# CBHFF
python train_cbhff.py    --dataset vitaldb --sample-wise --window-seconds 10   # RTTS
python train_cbhff.py    --dataset vitaldb --window-seconds 10                # STTS
python train_cbhff.py    --dataset vitaldb --pas                              # PAS
python train_cbhff.py    --dataset vitaldb                                    # LSPO

# CatBoost
python train_catboost.py --dataset vitaldb --sample-wise --window-seconds 10   # RTTS
python train_catboost.py --dataset vitaldb --window-seconds 10                # STTS
python train_catboost.py --dataset vitaldb --pas                              # PAS
python train_catboost.py --dataset vitaldb                                    # LSPO
```

### MUST

<table>
<thead>
<tr>
<th>Model</th><th>Split</th><th>MAE</th><th>RMSE</th><th>R2</th><th>Accuracy</th><th>Zone A</th><th>Zone B</th><th>Zone A+B</th><th>MARD</th><th>Pearson's r</th>
</tr>
</thead>
<tbody>
<tr><td rowspan="4">TinyML
  (ResNet34)</td><td>RTTS</td><td>0.7335 ± 0.0554</td><td>0.9909 ± 0.0789</td><td>0.0695 ± 0.0510</td><td>71.55 ± 3.92</td><td>85.82 ± 3.25</td><td>14.18 ± 3.25</td><td>100.00 ± 0.00</td><td>11.14 ± 0.83</td><td>0.3573 (p&lt;0.001)</td></tr>
<tr><td>STTS</td><td>0.8377 ± 0.1440</td><td>1.0576 ± 0.2645</td><td>-0.3119 ± 0.5170</td><td>65.80 ± 10.90</td><td>81.86 ± 9.36</td><td>18.14 ± 9.36</td><td>100.00 ± 0.00</td><td>12.87 ± 1.85</td><td>-0.2089 (p&lt;0.001)</td></tr>
<tr><td>PAS</td><td>0.8668 ± 0.2544</td><td>1.0350 ± 0.3052</td><td>-4.0480 ± 14.4014</td><td>61.11 ± 22.33</td><td>80.76 ± 18.87</td><td>19.24 ± 18.87</td><td>100.00 ± 0.00</td><td>13.74 ± 4.51</td><td>-0.1672 (p=0.1797)</td></tr>
<tr><td>LSPO</td><td>0.8963 ± 0.1748</td><td>1.0976 ± 0.2734</td><td>-0.4476 ± 0.6698</td><td>61.38 ± 14.65</td><td>76.92 ± 12.22</td><td>23.08 ± 12.22</td><td>100.00 ± 0.00</td><td>13.83 ± 2.63</td><td>-0.2654 (p=0.02999)</td></tr>
<tr><td rowspan="4">CBHFF</td><td>RTTS</td><td>0.7937 ± 0.0377</td><td>1.0327 ± 0.0751</td><td>-0.0104 ± 0.0192</td><td>70.48 ± 4.40</td><td>85.91 ± 3.17</td><td>14.09 ± 3.17</td><td>100.00 ± 0.00</td><td>12.03 ± 0.59</td><td>-0.0149 (p=0.7008)</td></tr>
<tr><td>STTS</td><td>0.8153 ± 0.1345</td><td>1.0267 ± 0.2690</td><td>-0.2234 ± 0.4748</td><td>68.39 ± 8.76</td><td>86.25 ± 7.94</td><td>13.75 ± 7.94</td><td>100.00 ± 0.00</td><td>12.41 ± 1.47</td><td>-0.3634 (p&lt;0.001)</td></tr>
<tr><td>PAS</td><td>0.7831 ± 0.1175</td><td>0.9713 ± 0.2493</td><td>-1.2585 ± 2.8590</td><td>71.23 ± 12.76</td><td>87.83 ± 10.57</td><td>12.17 ± 10.57</td><td>100.00 ± 0.00</td><td>12.16 ± 1.80</td><td>-0.2861 (p=0.01987)</td></tr>
<tr><td>LSPO</td><td>0.8096 ± 0.1454</td><td>1.0154 ± 0.2733</td><td>-0.1773 ± 0.3593</td><td>67.14 ± 11.52</td><td>86.15 ± 7.55</td><td>13.85 ± 7.55</td><td>100.00 ± 0.00</td><td>12.31 ± 1.67</td><td>-0.2784 (p=0.02253)</td></tr>
<tr><td rowspan="4">CatBoost</td><td>RTTS</td><td>0.5686 ± 0.0452</td><td>0.8986 ± 0.0924</td><td>0.2362 ± 0.0660</td><td>81.34 ± 3.04</td><td>90.69 ± 2.82</td><td>9.31 ± 2.82</td><td>100.00 ± 0.00</td><td>8.21 ± 0.50</td><td>0.5488 (p&lt;0.001)</td></tr>
<tr><td>STTS</td><td>0.8204 ± 0.2057</td><td>1.0984 ± 0.3130</td><td>-0.3468 ± 0.2814</td><td>64.64 ± 13.56</td><td>79.85 ± 9.88</td><td>20.15 ± 9.88</td><td>100.00 ± 0.00</td><td>11.97 ± 2.62</td><td>-0.0172 (p=0.6569)</td></tr>
<tr><td>PAS</td><td>0.7104 ± 0.1520</td><td>0.9182 ± 0.2699</td><td>-0.9315 ± 2.4096</td><td>71.62 ± 16.06</td><td>92.43 ± 7.82</td><td>7.57 ± 7.82</td><td>100.00 ± 0.00</td><td>10.78 ± 1.72</td><td>0.3124 (p=0.01067)</td></tr>
<tr><td>LSPO</td><td>0.7075 ± 0.1777</td><td>0.9474 ± 0.2811</td><td>0.0003 ± 0.2226</td><td>73.69 ± 11.89</td><td>89.43 ± 8.48</td><td>10.57 ± 8.48</td><td>100.00 ± 0.00</td><td>10.49 ± 2.16</td><td>0.3551 (p=0.003188)</td></tr>
</tbody>
</table>

```bash
# TinyML(ResNet34)
python train_resnet.py   --dataset MUST --sample-wise --window-seconds 1   # RTTS
python train_resnet.py   --dataset MUST --window-seconds 1                # STTS
python train_resnet.py   --dataset MUST --pas                             # PAS
python train_resnet.py   --dataset MUST                                   # LSPO

# CBHFF
python train_cbhff.py    --dataset MUST --sample-wise --window-seconds 1   # RTTS
python train_cbhff.py    --dataset MUST --window-seconds 1                # STTS
python train_cbhff.py    --dataset MUST --pas                             # PAS
python train_cbhff.py    --dataset MUST                                   # LSPO

# CatBoost
python train_catboost.py --dataset MUST --sample-wise --window-seconds 1   # RTTS
python train_catboost.py --dataset MUST --window-seconds 1                # STTS
python train_catboost.py --dataset MUST --pas                             # PAS
python train_catboost.py --dataset MUST                                   # LSPO
```

### IPS-PPG

PAS is `-` (one recording per person).

<table>
<thead>
<tr>
<th>Model</th><th>Split</th><th>MAE</th><th>RMSE</th><th>R2</th><th>Accuracy</th><th>Zone A</th><th>Zone B</th><th>Zone A+B</th><th>MARD</th><th>Pearson's r</th>
</tr>
</thead>
<tbody>
<tr><td rowspan="4">TinyML
  (ResNet34)</td><td>RTTS</td><td>0.8033 ± 0.0651</td><td>1.2021 ± 0.0993</td><td>0.2207 ± 0.0754</td><td>69.57 ± 5.45</td><td>77.66 ± 4.42</td><td>22.34 ± 4.42</td><td>100.00 ± 0.00</td><td>13.16 ± 1.11</td><td>0.5393 (p&lt;0.001)</td></tr>
<tr><td>STTS</td><td>0.8966 ± 0.1194</td><td>1.3252 ± 0.2055</td><td>0.0122 ± 0.2534</td><td>65.67 ± 7.94</td><td>74.02 ± 7.13</td><td>25.98 ± 7.13</td><td>100.00 ± 0.00</td><td>14.65 ± 2.22</td><td>0.3365 (p&lt;0.001)</td></tr>
<tr><td>PAS</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td></tr>
<tr><td>LSPO</td><td>1.0850 ± 0.1876</td><td>1.4247 ± 0.1642</td><td>-0.1421 ± 0.1843</td><td>46.32 ± 19.37</td><td>58.33 ± 17.42</td><td>41.67 ± 17.42</td><td>100.00 ± 0.00</td><td>18.54 ± 3.96</td><td>-0.0191 (p=0.7597)</td></tr>
<tr><td rowspan="4">CBHFF</td><td>RTTS</td><td>0.9017 ± 0.0871</td><td>1.2923 ± 0.1104</td><td>0.1010 ± 0.0736</td><td>62.22 ± 5.82</td><td>71.12 ± 4.65</td><td>28.88 ± 4.65</td><td>100.00 ± 0.00</td><td>14.97 ± 1.55</td><td>0.3964 (p&lt;0.001)</td></tr>
<tr><td>STTS</td><td>0.9417 ± 0.1341</td><td>1.3204 ± 0.1892</td><td>0.0283 ± 0.1322</td><td>59.75 ± 9.42</td><td>68.82 ± 8.05</td><td>31.18 ± 8.05</td><td>100.00 ± 0.00</td><td>15.83 ± 2.24</td><td>0.3570 (p&lt;0.001)</td></tr>
<tr><td>PAS</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td></tr>
<tr><td>LSPO</td><td>1.0437 ± 0.1279</td><td>1.3820 ± 0.1682</td><td>-0.0669 ± 0.1033</td><td>50.09 ± 12.12</td><td>62.49 ± 9.90</td><td>37.51 ± 9.90</td><td>100.00 ± 0.00</td><td>17.75 ± 2.41</td><td>0.1693 (p=0.006422)</td></tr>
<tr><td rowspan="4">CatBoost</td><td>RTTS</td><td>0.4360 ± 0.0569</td><td>0.8698 ± 0.1588</td><td>0.5745 ± 0.0843</td><td>91.46 ± 1.95</td><td>93.55 ± 1.53</td><td>6.45 ± 1.53</td><td>100.00 ± 0.00</td><td>6.57 ± 0.58</td><td>0.7833 (p&lt;0.001)</td></tr>
<tr><td>STTS</td><td>0.5945 ± 0.1252</td><td>1.0181 ± 0.2715</td><td>0.3925 ± 0.1319</td><td>85.71 ± 4.50</td><td>89.59 ± 4.29</td><td>10.41 ± 4.29</td><td>100.00 ± 0.00</td><td>9.18 ± 1.14</td><td>0.6420 (p&lt;0.001)</td></tr>
<tr><td>PAS</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td></tr>
<tr><td>LSPO</td><td>0.5786 ± 0.1285</td><td>0.9900 ± 0.2858</td><td>0.4285 ± 0.1197</td><td>85.16 ± 4.21</td><td>90.40 ± 4.26</td><td>9.60 ± 4.26</td><td>100.00 ± 0.00</td><td>8.97 ± 1.27</td><td>0.6765 (p&lt;0.001)</td></tr>
</tbody>
</table>

```bash
# TinyML(ResNet34)
python train_resnet.py   --dataset IPS --sample-wise --window-seconds 10   # RTTS
python train_resnet.py   --dataset IPS --window-seconds 10                # STTS
python train_resnet.py   --dataset IPS                                    # LSPO

# CBHFF
python train_cbhff.py    --dataset IPS --sample-wise --window-seconds 10   # RTTS
python train_cbhff.py    --dataset IPS --window-seconds 10                # STTS
python train_cbhff.py    --dataset IPS                                    # LSPO

# CatBoost
python train_catboost.py --dataset IPS --sample-wise --window-seconds 10   # RTTS
python train_catboost.py --dataset IPS --window-seconds 10                # STTS
python train_catboost.py --dataset IPS                                    # LSPO
```

## Table 6 (OOD)

Single-split results (no std). Order matches the paper: VitalDB→MUST, IPS-PPG→MUST, VitalDB→IPS-PPG, IPS-PPG→VitalDB. MUST→VitalDB and MUST→IPS-PPG are omitted from the paper (small source set) but included here. MAE and RMSE are in mmol/L; Accuracy, Zone A/B/A+B, and MARD are in %.

<table>
<thead>
<tr>
<th>Train → test</th><th>Model</th><th>MAE</th><th>RMSE</th><th>R2</th><th>Accuracy</th><th>Zone A</th><th>Zone B</th><th>Zone A+B</th><th>MARD</th><th>Pearson's r</th>
</tr>
</thead>
<tbody>
<tr><td rowspan="3">VitalDB → MUST</td><td>TinyML(ResNet34)</td><td>1.0642</td><td>1.2436</td><td>-0.4491</td><td>43.28</td><td>56.72</td><td>43.28</td><td>100</td><td>17.54</td><td>-0.1275</td></tr>
<tr><td>CBHFF</td><td>1.0463</td><td>1.2261</td><td>-0.4085</td><td>38.81</td><td>64.18</td><td>35.82</td><td>100</td><td>17.24</td><td>0.0933</td></tr>
<tr><td>CatBoost</td><td>0.7997</td><td>1.1511</td><td>-0.2414</td><td>67.16</td><td>79.1</td><td>20.9</td><td>100</td><td>11.34</td><td>0.1601</td></tr>
<tr><td rowspan="3">IPS-PPG → MUST</td><td>TinyML(ResNet34)</td><td>0.7448</td><td>1.1156</td><td>-0.1661</td><td>73.13</td><td>86.57</td><td>13.43</td><td>100</td><td>10.62</td><td>0.2099</td></tr>
<tr><td>CBHFF</td><td>0.7723</td><td>1.1262</td><td>-0.1883</td><td>65.67</td><td>86.57</td><td>13.43</td><td>100</td><td>11.04</td><td>0.1075</td></tr>
<tr><td>CatBoost</td><td>0.9087</td><td>1.3144</td><td>-0.6187</td><td>62.69</td><td>68.66</td><td>31.34</td><td>100</td><td>12.7</td><td>0.1288</td></tr>
<tr><td rowspan="3">VitalDB → IPS-PPG</td><td>TinyML(ResNet34)</td><td>1.2318</td><td>1.5083</td><td>-0.2148</td><td>38.37</td><td>48.84</td><td>51.16</td><td>100</td><td>22.14</td><td>0.1402</td></tr>
<tr><td>CBHFF</td><td>1.3406</td><td>1.5517</td><td>-0.2858</td><td>24.03</td><td>35.27</td><td>64.73</td><td>100</td><td>24.72</td><td>0.2518</td></tr>
<tr><td>CatBoost</td><td>0.9432</td><td>1.2101</td><td>0.1907</td><td>53.17</td><td>62.7</td><td>37.3</td><td>100</td><td>16.81</td><td>0.6102</td></tr>
<tr><td rowspan="3">IPS-PPG → VitalDB</td><td>TinyML(ResNet34)</td><td>1.5717</td><td>2.0406</td><td>-0.0515</td><td>36.38</td><td>47.3</td><td>50.13</td><td>97.43</td><td>25.16</td><td>0.0651</td></tr>
<tr><td>CBHFF</td><td>1.4689</td><td>2.0183</td><td>-0.0287</td><td>41.16</td><td>53.85</td><td>43.57</td><td>97.43</td><td>22.25</td><td>0.0027</td></tr>
<tr><td>CatBoost</td><td>1.4991</td><td>1.9614</td><td>0.0285</td><td>37.68</td><td>50.03</td><td>47.39</td><td>97.43</td><td>23.84</td><td>0.217</td></tr>
<tr><td rowspan="3">MUST → VitalDB</td><td>TinyML(ResNet34)</td><td>1.4223</td><td>2.0348</td><td>-0.0456</td><td>44.61</td><td>56.77</td><td>40.65</td><td>97.43</td><td>20.61</td><td>-0.0567</td></tr>
<tr><td>CBHFF</td><td>1.4194</td><td>2.0217</td><td>-0.0321</td><td>43.95</td><td>55.91</td><td>41.52</td><td>97.43</td><td>20.7</td><td>-0.0436</td></tr>
<tr><td>CatBoost</td><td>1.4823</td><td>2.0023</td><td>-0.0124</td><td>39.67</td><td>51.84</td><td>45.58</td><td>97.43</td><td>22.85</td><td>-0.0219</td></tr>
<tr><td rowspan="3">MUST → IPS-PPG</td><td>TinyML(ResNet34)</td><td>1.2772</td><td>1.4727</td><td>-0.372</td><td>23.12</td><td>35.84</td><td>64.16</td><td>100</td><td>23.91</td><td>-0.0504</td></tr>
<tr><td>CBHFF</td><td>1.3394</td><td>1.522</td><td>-0.4654</td><td>20.23</td><td>33.53</td><td>66.47</td><td>100</td><td>25.29</td><td>-0.0208</td></tr>
<tr><td>CatBoost</td><td>1.4741</td><td>1.6418</td><td>-0.49</td><td>18.25</td><td>28.17</td><td>71.83</td><td>100</td><td>27.53</td><td>0.1772</td></tr>
</tbody>
</table>

```bash
# VitalDB → MUST
python train_resnet_cross.py   --train-dataset vitaldb --test-dataset MUST
python train_cbhff_cross.py    --train-dataset vitaldb --test-dataset MUST
python train_catboost_cross.py --train-dataset vitaldb --test-dataset MUST

# IPS-PPG → MUST
python train_resnet_cross.py   --train-dataset IPS --test-dataset MUST
python train_cbhff_cross.py    --train-dataset IPS --test-dataset MUST
python train_catboost_cross.py --train-dataset IPS --test-dataset MUST

# VitalDB → IPS-PPG
python train_resnet_cross.py   --train-dataset vitaldb --test-dataset IPS
python train_cbhff_cross.py    --train-dataset vitaldb --test-dataset IPS
python train_catboost_cross.py --train-dataset vitaldb --test-dataset IPS

# IPS-PPG → VitalDB
python train_resnet_cross.py   --train-dataset IPS --test-dataset vitaldb
python train_cbhff_cross.py    --train-dataset IPS --test-dataset vitaldb
python train_catboost_cross.py --train-dataset IPS --test-dataset vitaldb

# MUST → VitalDB
python train_resnet_cross.py   --train-dataset MUST --test-dataset vitaldb
python train_cbhff_cross.py    --train-dataset MUST --test-dataset vitaldb
python train_catboost_cross.py --train-dataset MUST --test-dataset vitaldb

# MUST → IPS-PPG
python train_resnet_cross.py   --train-dataset MUST --test-dataset IPS
python train_cbhff_cross.py    --train-dataset MUST --test-dataset IPS
python train_catboost_cross.py --train-dataset MUST --test-dataset IPS
```

Smoke test: `python train_resnet.py --dataset MUST --quick --max-samples 16`
