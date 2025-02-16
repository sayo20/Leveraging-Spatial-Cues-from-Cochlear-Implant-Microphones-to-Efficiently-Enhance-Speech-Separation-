# Leveraging-Spatial-Cues-from-Cochlear-Implant-Microphones-to-Efficiently-Enhance-Speech-Separation-

# Abstract
While speech separation approaches for single-channel, dry speech mixtures have improved greatly, speech separation in real-world, spatial and reverberant acoustic environments remains challenging. This limits the efficiency of existing approaches for real-world speech separation applications in assistive hearing devices such as cochlear implants (CIs). To address this issue, we investigated to what extent incorporating spatial cues captured by CI microphones can improve speech separation quality in an efficient manner. We analyze the impact on speech separation performance of implicit spatial cues (i.e., cues inherently present in multi-channel data that can be learned by the model), as well as of explicit spatial cues (i.e., spatial features such as interaural level differences [ILD] and interaural phase differences [IPD] which are added as auxiliary input to the model). Our findings demonstrate that implicit spatial cues can enhance speech perception without affecting model latency and efficiency, while explicit spatial cues enhance speech perception most but at the same time substantially reduce model efficiency. Of the explicit spatial cues, IPD enhanced speech separation significantly more than ILD or a combination of IPD and ILD. Moreover, IPD improved separation performance both for spatially distant and spatially overlapping talkers, whereas implicit spatial cues only enhanced separation performance for spatially distant talkers. Both implicit and explicit spatial cues enhanced speech separation in particular when spectral cues for separation are ambiguous, that is, when voices are similar. CI simulations demonstrated that incorporating spatial cues enhances speech perception in CI users in a similar manner as speech perception in normal hearing listeners. In sum, these findings elucidate the benefits and costs of incorporating implicit and explicit spatial cues into speech separation frameworks, thereby contributing to the development of more efficient speech separation approaches for real-world applications such as CIs.

# Using checkpoints
We have uploaded the training checkpoints for the different input configuration used in this study. They can be found here: https://drive.google.com/drive/folders/18l62J_Xon2zpKJU1mLcMzkx8FzBP0nUh?usp=sharing

# Training
To train from scratch, you can run the RunTrain file. you need to switch to either LightningFile or LighningFIle_spatial dependending on what input configuration you are using.

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
      <audio controls>
        <source src="path/to/mixed_audio1.wav" type="audio/wav">
        Your browser does not support the audio element.
      </audio>
    </td>
    <td>
      <audio controls>
        <source src="path/to/reference_audio1.wav" type="audio/wav">
        Your browser does not support the audio element.
      </audio>
    </td>
    <td>
      <audio controls>
        <source src="path/to/separated_audio1.wav" type="audio/wav">
        Your browser does not support the audio element.
      </audio>
    </td>
  </tr>
  <tr>
    <td>
      <audio controls>
        <source src="path/to/mixed_audio2.wav" type="audio/wav">
        Your browser does not support the audio element.
      </audio>
    </td>
    <td>
      <audio controls>
        <source src="path/to/reference_audio2.wav" type="audio/wav">
        Your browser does not support the audio element.
      </audio>
    </td>
    <td>
      <audio controls>
        <source src="path/to/separated_audio2.wav" type="audio/wav">
        Your browser does not support the audio element.
      </audio>
    </td>
  </tr>
</table>

