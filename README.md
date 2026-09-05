# BCI Robot Control System

## Project Engineer

Surafel Gashaw (Leo Vance Hendrix)

## What this project is

This is a brain-computer interface system that reads real EEG signals from a
person's scalp, recognizes patterns in that signal that correspond to specific
imagined movements, and uses those patterns to drive a physical robot over a
serial connection. The system does not read thoughts and does not guess at
arbitrary mental states. It recognizes a small, fixed set of trained patterns
for one specific person, based on calibration data that person provides
themselves. Nothing about the classifier's output is meaningful for anyone who
did not sit through the calibration session that trained it.

The pipeline, in order, is:

```
EEG headset
  -> EEG acquisition (BrainFlow)
  -> Signal quality check
  -> Preprocessing (detrend, band-pass filter, notch filter, normalize)
  -> Feature extraction (Common Spatial Patterns + log-variance, or band power)
  -> Classifier (LDA / SVM / Logistic Regression / Random Forest)
  -> Confidence threshold and temporal smoothing
  -> Safety controller (mandatory gate)
  -> Robot controller (serial protocol)
  -> Robot
```

The safety controller sits between the classifier and the robot at all times.
The classifier is never allowed to talk to the robot directly. If the signal
is bad, if confidence is low, if the EEG board disconnects, if the robot stops
acknowledging commands, or if the emergency stop is triggered, the robot is
forced to stop, no matter what the classifier said.

## Libraries used

This project is written in Python 3.11 and depends on the following:

```
numpy>=1.24
scipy>=1.10
mne>=1.5
brainflow>=5.10.0
scikit-learn>=1.3
pyserial>=3.5
pandas>=2.0
matplotlib>=3.7
joblib>=1.3
opencv-python>=4.8
```

Install them with:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Here is what each library is actually responsible for in this codebase, so it
is clear why each one is there:

- **numpy** handles every array operation in the pipeline: EEG windows,
  covariance matrices, feature vectors.
- **scipy** provides the filter design and filtering functions (Butterworth
  band-pass, IIR notch, Welch's method for power spectral density) used in
  `eeg/preprocessing.py` and `eeg/quality.py`.
- **mne** provides the Common Spatial Patterns implementation used in
  `features/csp.py`. This is a well-validated, widely used implementation
  rather than something written from scratch, because CSP has enough subtle
  numerical detail (whitening, eigenvalue ordering) that it is not worth
  reimplementing.
- **brainflow** is the hardware abstraction layer for the EEG device itself,
  used in `eeg/acquisition.py`. It is what actually talks to the EEG board,
  handles sampling rate configuration, and exposes a synthetic board used
  strictly for software testing.
- **scikit-learn** provides the classifiers (LDA, linear SVM, logistic
  regression, random forest) and the cross-validation and scoring utilities
  used in `models/classifier.py`.
- **pyserial** is the serial communication library used in
  `robot/controller.py` to talk to the Arduino or ESP32 controlling the
  robot's motors.
- **pandas** is used for handling tabular data during training and evaluation,
  such as organizing epochs, labels, and metrics.
- **matplotlib** is available for plotting EEG waveforms, band power, and
  training results, for anyone inspecting a session outside of the live
  dashboard.
- **joblib** is used to serialize and load the trained classifier and
  preprocessing objects in `models/model_manager.py`.
- **opencv-python** is included for any future visual feedback or camera-based
  monitoring of the robot; it is not required by the core EEG-to-robot
  pipeline itself, but is part of the dependency list per the original
  specification for this project.

## Hardware setup

1. **EEG device.** Any board supported by BrainFlow works, including OpenBCI
   Cyton, OpenBCI Cyton with Daisy, g.tec Unicorn, or a Muse headset through a
   BrainFlow-compatible bridge. Look up the integer board ID for your specific
   hardware in BrainFlow's documentation and set it in `config.py`.
2. **Electrode placement.** For motor imagery classification (left hand,
   right hand, feet), place electrodes at minimum over C3, C4, and Cz using
   the 10-20 system, with additional surrounding channels if your headset
   supports them. Check electrode impedance before every session; a poor
   connection will fail the signal quality check before it ever reaches the
   classifier.
3. **Robot.** An Arduino or ESP32 running the firmware in
   `firmware/bci_robot_receiver.ino`, connected to a basic two-motor
   differential drive through a motor driver such as an L298N or TB6612.
   Adjust the pin numbers in the firmware to match your wiring, flash it, and
   note the serial port it appears on.

## Calibration

A classifier trained on someone else's brain will not work reliably on a
different person. Before training, the intended user must go through a
calibration session:

```bash
python main.py collect --simulation    # dry run using BrainFlow's synthetic board
python main.py collect                  # real EEG, using the board configured in config.py
```

This walks the user through repeated trials: rest, imagine left hand, rest,
imagine right hand, rest, imagine feet, and so on, and saves the recorded
epochs to `data/calibration_session.npz`. At least twenty to thirty trials per
class is a reasonable minimum; more trials generally produce a more reliable
classifier.

## Training

```bash
python main.py train --session data/calibration_session.npz --out trained_models
```

Training rejects contaminated trials, filters and epochs the remaining data,
extracts features, cross-validates several classifiers, selects the one with
the best held-out performance, and prints a full report. If the resulting
accuracy is too low to be usable, the program says so directly rather than
hiding it.

## Real-time control

```bash
python main.py run --simulation                            # simulated EEG, no real robot
python main.py run --robot-port /dev/ttyUSB0 --simulation    # simulated EEG, real robot
python main.py run --robot-port /dev/ttyUSB0                 # real EEG, real robot
```

Leaving out `--robot-port` runs the robot side as a dry run that only logs
what it would have sent. A dashboard built with Tkinter shows EEG connection
state, signal quality, the predicted command, confidence, the robot's current
state, a live waveform, latency broken down by stage, and emergency stop
state. Pressing the space bar or clicking the on-screen button triggers an
immediate emergency stop that overrides every other part of the system,
including a high-confidence classifier output.

## Testing

```bash
pip install pytest
pytest tests/
```

Tests cover preprocessing, signal quality checks, temporal smoothing, the
serial protocol, and the safety controller's override behavior, all using
fake data and fake hardware stand-ins so no physical hardware is required to
run them.

## Project structure

```
bci_robot/
  main.py                    entry point: collect, train, run
  config.py                  all configuration values, thresholds, mappings
  realtime_bci.py            real-time loop that wires everything together
  eeg/
    acquisition.py           BrainFlow connection, real hardware and simulation
    preprocessing.py         detrend, band-pass, notch, normalize
    quality.py               signal quality scoring
    artifacts.py             epoch-level artifact rejection for training data
  features/
    csp.py                   Common Spatial Patterns and log-variance
    bandpower.py              absolute and relative band power
  models/
    classifier.py             classifier training, cross-validation, selection
    model_manager.py           saving and loading trained models
  training/
    collect_data.py            calibration session recording
    train_bci.py                full training pipeline
    evaluate.py                  validation report printing
  realtime/
    predictor.py                real-time prediction pipeline
    smoothing.py                 confidence threshold and temporal voting
  robot/
    controller.py                 serial connection, acknowledgements, heartbeat
    serial_protocol.py             wire protocol definitions
  safety/
    safety_controller.py           the mandatory gate between classifier and robot
  gui/
    dashboard.py                    real-time dashboard
  firmware/
    bci_robot_receiver.ino           Arduino and ESP32 firmware
  tests/                              unit tests
  data/                                calibration sessions are saved here
  trained_models/                      trained model files are saved here
```

## The mathematics behind the pipeline

This section explains the actual formulas used at each stage. Nothing here is
decorative; every equation below corresponds directly to code in this
repository.

### EEG sampling

Each electrode records a continuous voltage over time. The EEG board samples
that continuous signal at a fixed rate, producing a discrete sequence:

x_c[n] = x_c(n / f_s)

where x_c(t) is the continuous voltage at channel c, f_s is the sampling rate
in hertz, and n is the sample index. At any point in time, the system holds a
window of the most recent W seconds across all C channels as a matrix:

X is a C by L matrix, where L = floor(W times f_s)

L is the number of samples per channel inside that window. This matrix is
what moves through every stage described below.

### Detrending

A straight line is fit to each channel by least squares and subtracted, to
remove slow drift that is not part of the EEG rhythms of interest:

x'[n] = x[n] - (a_hat times n + b_hat)

where (a_hat, b_hat) are the slope and intercept that minimize the sum of
squared differences between the fitted line and the signal:

(a_hat, b_hat) = argmin over (a, b) of the sum over n of (x[n] - a*n - b) squared

### Band-pass filtering

EEG rhythms relevant to motor imagery sit roughly between 1 Hz and 40 Hz. A
Butterworth filter is used because its passband has no ripple, which matters
because ripple would distort the relative power comparisons between bands
later on. Its magnitude response, for a low-pass prototype of order N with
cutoff f_c, is:

magnitude squared of H(f) = 1 divided by (1 plus (f_c divided by f) to the power of 2N)

The cutoff frequencies are normalized against the Nyquist frequency, which is
half the sampling rate, before the filter coefficients are computed:

f_low_normalized = f_low / (f_s / 2)
f_high_normalized = f_high / (f_s / 2)

Once the coefficients a and b are designed, the filter is applied as a linear
difference equation relating output y to input x:

sum over k of a_k times y[n-k] = sum over k of b_k times x[n-k]

### Notch filtering for power line interference

Electrical mains interference shows up as a narrow spike at 50 Hz or 60 Hz. A
notch filter removes that specific frequency and leaves everything around it
alone. How narrow the removed band is depends on the quality factor Q:

Q = f_0 / delta_f

where f_0 is the center frequency being removed and delta_f is the width of
the band being suppressed around it. A higher Q removes a narrower slice of
spectrum.

### Causal filtering versus zero-phase filtering

During training, the entire recording already exists, so filtering can be
done both forward and backward through the signal, which cancels out any
phase distortion the filter would otherwise introduce:

Y(z) = H(z) times H(1/z) times X(z)

This is not possible in real time, because filtering backward through a
signal requires samples that have not been recorded yet. In real-time
operation, only the forward, causal version of the difference equation is
used, and the filter's internal delay-line state is kept and carried forward
from one window to the next, so filtering stays continuous instead of
resetting at the start of every new chunk of data.

### Normalization

Each channel is rescaled to zero mean and unit variance, so that differences
in electrode contact quality or amplifier gain between channels do not
distort the features that come later:

x_normalized[n] = (x[n] - mu) / sigma

where mu is the mean of the channel and sigma is its standard deviation.

### Signal quality: variance

Per-channel variance is computed directly and checked against both a lower
and an upper bound:

Var(x) = (1/N) times the sum over n of (x[n] - mu) squared

Variance far below the expected range suggests a disconnected electrode.
Variance far above the expected range suggests saturation, movement artifact,
or a poor connection picking up external noise.

### Signal quality: power spectral density

To detect low-frequency contamination such as eye blinks, the power spectral
density is estimated using Welch's method. The window is split into
overlapping segments, each segment is windowed and transformed, and the
results are averaged together, which reduces the variance of the estimate
compared to a single periodogram computed over the whole window:

S_xx(f) = (1/K) times the sum over k of the squared magnitude of the Fourier
transform of (w_k times x_k)

where x_k is the k-th overlapping segment, w_k is a window function such as a
Hann window applied to reduce spectral leakage, and K is the number of
segments being averaged.

### Signal quality: band power ratio

The fraction of a window's total power that falls inside a specific frequency
band is:

ratio = (sum of S_xx(f) for f between f1 and f2) divided by (sum of S_xx(f)
over all f)

If the ratio of power in the low-frequency band associated with eye blinks
exceeds a configured threshold, the window is flagged as likely contaminated
and rejected before it reaches the classifier.

### Band power features

For each of the five standard EEG bands, delta, theta, alpha, beta, and
gamma, absolute band power is computed by integrating the power spectral
density over that band's frequency range. Because the PSD is only known at
discrete frequency bins from the Fourier transform, the integral is
approximated as a sum:

P_band = the integral from f1 to f2 of S_xx(f) df, approximated as the sum
over f between f1 and f2 of S_xx(f) times delta_f

Relative band power divides this by the total power across all bands, which
makes the feature less sensitive to a person's overall signal amplitude and
more sensitive to the relative balance between bands, which is what changes
during different mental tasks.

### Common Spatial Patterns

Common Spatial Patterns, referred to as CSP, is the central feature
extraction method for motor imagery classification in this project. Rather
than looking at each electrode independently, CSP finds linear combinations
of electrodes, called spatial filters, whose variance differs as much as
possible between two classes, for example left-hand imagery against
right-hand imagery.

For each class, a spatial covariance matrix is estimated from the training
epochs of that class:

Sigma_class = (1 / (L - 1)) times X_class times X_class transposed, averaged
over all epochs belonging to that class

CSP then solves a generalized eigenvalue problem:

Sigma_1 times w = lambda times (Sigma_1 + Sigma_2) times w

The eigenvectors w associated with the largest and smallest eigenvalues
lambda are kept as spatial filters. These particular eigenvectors maximize a
Rayleigh quotient, which is the ratio of one class's variance to the combined
variance of both classes:

lambda = (w transposed times Sigma_1 times w) divided by (w transposed times
(Sigma_1 plus Sigma_2) times w)

Once a new epoch is projected through a spatial filter, the log of its
variance becomes a single feature:

f_i = log( Var(w_i transposed times X) divided by the sum over j of
Var(w_j transposed times X) )

These log-variance values, typically four to eight numbers per epoch instead
of raw values across every channel and every timepoint, are what the
classifier actually receives. Because standard CSP is a two-class method,
more than two classes are handled with a one-versus-rest extension: one CSP
is trained per class against everything else, and the resulting feature sets
are concatenated together.

### Linear Discriminant Analysis

Linear Discriminant Analysis finds the linear projection that best separates
classes by maximizing the ratio of between-class variance to within-class
variance, known as Fisher's criterion:

J(w) = (w transposed times S_B times w) divided by (w transposed times S_W
times w)

where S_B is the between-class scatter matrix and S_W is the within-class
scatter matrix. In the two-class case, the resulting decision rule is a
simple linear boundary:

y_hat = sign(w transposed times x + b)

### Support Vector Machine

The linear support vector machine finds the separating hyperplane that
maximizes the margin, meaning the distance from the hyperplane to the closest
training points on either side:

minimize (1/2) times the squared norm of w, subject to y_i times (w
transposed times x_i + b) being greater than or equal to 1 for every training
point i

A linear kernel is used rather than a nonlinear one, because CSP features are
already a compact and discriminative representation, and a linear boundary is
less likely to overfit given the relatively small number of trials a
calibration session produces.

### Logistic Regression

Logistic regression estimates class probability directly using the sigmoid
function:

P(y = 1 given x) = sigma(w transposed times x + b) = 1 divided by (1 plus e
to the power of negative (w transposed times x + b))

Its parameters are fit by minimizing cross-entropy loss over the training
data:

Loss = negative (1/N) times the sum over i of [ y_i times log(p_hat_i) plus
(1 - y_i) times log(1 - p_hat_i) ]

### Random Forest

A random forest is an ensemble of decision trees, each trained on a
bootstrap-resampled subset of the training epochs and a random subset of
features considered at each split. Its final prediction is the majority vote,
or the averaged class probability, across all T trees in the forest:

P(y = k given x) = (1/T) times the sum over t of an indicator function equal
to 1 if tree_t predicts class k for x, and 0 otherwise

A random forest can capture nonlinear relationships between features that a
linear model cannot, at the cost of generally needing more training data to
avoid overfitting.

### Model selection

All four classifiers are evaluated using the same cross-validation split, and
the one with the highest balanced accuracy on held-out data is refit on the
full dataset and saved as the final model. This selection is never based on
training accuracy, because training accuracy measures how well a model
memorized the data it was trained on, not how well it will perform on new
data.

### Cross-validation

Because a calibration session only produces a limited number of trials,
k-fold cross-validation is used. The data is split into k folds. Each fold is
held out once as a test set while the model is trained on the remaining k
minus 1 folds. Every reported metric is computed only on predictions made on
data the model never saw during that particular training run.

### Accuracy and balanced accuracy

Accuracy is the simplest measure:

Accuracy = (number of correct predictions) divided by (total number of
predictions)

Balanced accuracy averages recall across classes instead of pooling all
predictions together, which matters because a calibration session rarely
produces perfectly equal numbers of trials per class:

Balanced Accuracy = (1/K) times the sum over k of Recall for class k

### Precision, recall, and F1 score

Precision = true positives divided by (true positives plus false positives)
Recall = true positives divided by (true positives plus false negatives)
F1 = 2 times (Precision times Recall) divided by (Precision plus Recall)

These are macro-averaged across all classes, meaning every class contributes
equally to the final number regardless of how many trials it has.

### Confusion matrix

A K by K matrix where the entry at row i, column j counts how many trials
whose true class was i were predicted as class j. The diagonal shows correct
predictions. Everything off the diagonal shows exactly which classes are
being confused with which other classes, which a single accuracy number
cannot reveal on its own.

### Confidence and temporal smoothing

At each moment, the classifier outputs a probability for every class. The
predicted command is the class with the highest probability, and confidence
is that probability itself:

command = the class k that maximizes P(y = k given x)
confidence = the maximum value of P(y = k given x) over all k

If confidence falls below a configured threshold, for example 0.80, the
system does not act on the prediction at all. It treats that moment as
ambiguous rather than guessing.

A single window's prediction is never sent to the robot on its own. The
system requires the same command to appear consistently across M consecutive
high-confidence windows before it is allowed through:

emit at time t equals command at time t, only if command at time (t minus i)
equals command at time t for every i from 0 to M minus 1

emit at time t equals STOP in every other case

This is a straightforward streak counter over recent predictions rather than
a statistical model, but it is what turns a noisy, per-window classifier
output into something stable enough to safely control a physical robot. The
streak resets immediately whenever confidence drops or the predicted class
changes.

### Safety controller logic

The safety controller's job is logical rather than probabilistic. If any one
of the following conditions is true, the command sent to the robot is forced
to STOP, regardless of what the classifier and the smoothing stage produced:

- signal quality score is below threshold, or a hard failure such as
  saturation or disconnect was flagged
- classifier confidence is below the minimum threshold
- the required number of consistent windows has not yet been reached
- no new prediction has arrived within the maximum allowed command age
- the EEG board or the robot's serial connection has disconnected
- the robot failed to acknowledge the last command within its timeout window
- an invalid or unrecognized command was produced
- the emergency stop has been triggered, which always overrides everything
  else with no exception

Written as a single rule, if any of conditions C1 through C8 is true:

command_sent = STOP, if (C1 or C2 or ... or C8) is true, otherwise
command_sent = command_smoothed

The Arduino or ESP32 firmware enforces a second, independent version of this
same idea: if it does not receive a valid command within its own timeout
window, it stops the motors on its own, so a failure in the host computer's
software cannot leave the robot moving.

### Latency

Total end-to-end latency, from a sample being acquired to a command reaching
the robot, is the sum of every stage's processing time:

T_total = T_acquisition + T_preprocessing + T_classification +
T_communication

Each of these terms is measured directly with timestamps rather than
estimated, and shown live on the dashboard. Because the classifier only ever
sees a full window of W seconds at a time, T_acquisition has a hard floor of
roughly W seconds. Shortening the window reduces latency, but it also reduces
how much data each classification decision is based on, which usually costs
some accuracy. This is a genuine trade-off between speed and reliability, not
something that can be tuned away in either direction for free.

## A final note on what this system is and is not

This system does not read thoughts. It is a supervised pattern classifier
trained on a small number of predefined mental tasks, such as imagining
moving a hand or a foot, for one specific person. Its output reflects
statistical patterns learned from that person's own EEG during those
specific tasks in a specific calibration session, nothing more. Any accuracy
number reported by this system is specific to the person and session it was
measured on, and should be treated that way.
