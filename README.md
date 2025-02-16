# Leveraging-Spatial-Cues-from-Cochlear-Implant-Microphones-to-Efficiently-Enhance-Speech-Separation-

# Abstract
While speech separation approaches for single-channel, dry speech mixtures have improved greatly, speech separation in real-world, spatial and reverberant acoustic environments remains challenging. This limits the efficiency of existing approaches for real-world speech separation applications in assistive hearing devices such as cochlear implants (CIs). To address this issue, we investigated to what extent incorporating spatial cues captured by CI microphones can improve speech separation quality in an efficient manner. We analyze the impact on speech separation performance of implicit spatial cues (i.e., cues inherently present in multi-channel data that can be learned by the model), as well as of explicit spatial cues (i.e., spatial features such as interaural level differences [ILD] and interaural phase differences [IPD] which are added as auxiliary input to the model). Our findings demonstrate that implicit spatial cues can enhance speech perception without affecting model latency and efficiency, while explicit spatial cues enhance speech perception most but at the same time substantially reduce model efficiency. Of the explicit spatial cues, IPD enhanced speech separation significantly more than ILD or a combination of IPD and ILD. Moreover, IPD improved separation performance both for spatially distant and spatially overlapping talkers, whereas implicit spatial cues only enhanced separation performance for spatially distant talkers. Both implicit and explicit spatial cues enhanced speech separation in particular when spectral cues for separation are ambiguous, that is, when voices are similar. CI simulations demonstrated that incorporating spatial cues enhances speech perception in CI users in a similar manner as speech perception in normal-hearing listeners. In sum, these findings elucidate the benefits and costs of incorporating implicit and explicit spatial cues into speech separation frameworks, thereby contributing to the development of more efficient speech separation approaches for real-world applications such as CIs.

# Using checkpoints
We have uploaded the training checkpoints for the different input configurations used in this study. They can be found here: https://drive.google.com/drive/folders/18l62J_Xon2zpKJU1mLcMzkx8FzBP0nUh?usp=sharing

# Training
To train from scratch, you can run the RunTrain file. you need to switch to either LightningFile or LighningFIle_spatial depending on what input configuration you are using.

# Audio samples
## Audio Samples





<table>
  <tr>
    <th>Mixed Audio</th>
    <th>Reference Audio</th>
    <th>Separated Audio</th>
  </tr>
  <tr>
    <td>
Speakers at angles 90 and 270

      
  https://github.com/user-attachments/assets/521ee8aa-c506-4905-b859-d9dba5604476
    </td>
    <td>
    
https://github.com/user-attachments/assets/fe4e1ccc-c81a-42f9-97be-df23ebd5ef6c


      
https://github.com/user-attachments/assets/b9e0cf9f-2684-4e4f-82f9-1d3ba6c31b64
    </td>
    <td>
    
https://github.com/user-attachments/assets/5da2f37e-5986-428d-b940-dc054b97e27e

      
https://github.com/user-attachments/assets/9db1ee15-dee8-4f85-a45b-711033c64334
    </td>
  </tr>

  <tr>
    <td>
Speakers at angle 0

      
   https://github.com/user-attachments/assets/79fc1590-bb48-4db9-98d1-a390304c296d
    </td>
    <td>

https://github.com/user-attachments/assets/e8de39ea-d070-4cc6-9c16-9458736a1de6




https://github.com/user-attachments/assets/a3519ff4-7a95-43e6-a07d-9c4bf9e642a4
</td>
<td>

      
https://github.com/user-attachments/assets/22f525a4-2f5d-4466-9f34-44ad190cfd0c


https://github.com/user-attachments/assets/976e9fb8-9376-4c7e-82f7-ba9c91401f44
 </td>
  </tr>

  <tr>
    <td>
Speakers at angles 15 and 30

      

https://github.com/user-attachments/assets/727ae443-037c-43c0-a487-d09110f091b9
    </td>
    <td>

https://github.com/user-attachments/assets/c9e2cfa8-844e-46c3-99f2-605f70a9618c





https://github.com/user-attachments/assets/cc3e7ac1-eaba-4b9f-b998-eb3e1f28d74d
</td>
<td>

      
https://github.com/user-attachments/assets/bb7c374b-9bc4-4f2a-9455-91878f77c045



https://github.com/user-attachments/assets/6b110933-8186-401a-8a8e-ac2d427e970a
 </td>
  </tr>
</table>


