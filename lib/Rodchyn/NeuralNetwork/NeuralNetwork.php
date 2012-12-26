<?php

namespace Rodchyn\Neuralnetwork;

/**
 * Multi-layer Neural Network in PHP
 * 
 * Loosely based on source code by {@link http://www.philbrierley.com Phil Brierley},
 * that was translated into PHP by 'dspink' in sep 2005
 * 
 * Algorithm was obtained from the excellent introductory book 
 * "{@link http://www.amazon.com/link/dp/0321204662 Artificial Intelligence - a guide to intelligent systems}"
 * by Michael Negnevitsky (ISBN 0-201-71159-1)
 *
 * <b>Example: learning the 'XOR'-function</b>
 * <code>
 * require_once("class_neuralnetwork.php");
 * 
 * // Create a new neural network with 3 input neurons, 
 * // 4 hidden neurons, and 1 output neuron
 * $n = new NeuralNetwork(3, 4, 1);
 * $n->setVerbose(false);
 * 
 * // Add test-data to the network. In this case, 
 * // we want the network to learn the 'XOR'-function.
 * // The third input-parameter is the 'bias'.
 * $n->addTestData( array (-1, -1, 1), array (-1));
 * $n->addTestData( array (-1,  1, 1), array ( 1));
 * $n->addTestData( array ( 1, -1, 1), array ( 1));
 * $n->addTestData( array ( 1,  1, 1), array (-1));
 * 
 * // we try training the network for at most $max times
 * $max = 3;
 * 
 * // train the network in max 1000 epochs, with a max squared error of 0.01
 * while (!($success=$n->train(1000, 0.01)) && $max-->0) {
 *         // training failed:
 *         // 1. re-initialize the weights in the network
 *         $n->initWeights();
 *        
 *         // 2. display message
 *         echo "Nothing found...<hr />"; 
 * }
 * 
 * // print a message if the network was succesfully trained
 * if ($success) {
 *         $epochs = $n->getEpoch();
 *         echo "Success in $epochs training rounds!<hr />";
 * }
 * 
 * // in any case, we print the output of the neural network
 * for ($i = 0; $i < count($n->trainInputs); $i ++) {
 *         $output = $n->calculate($n->trainInputs[$i]);
 *         print "<br />Testset $i; ";
 *         print "expected output = (".implode(", ", $n->trainOutput[$i]).") ";
 *         print "output from neural network = (".implode(", ", $output).")\n";
 * }
 * </code>
 * 
 * The resulting output could for example be something along the following lines:
 * 
 * <code>
 * Success in 719 training rounds!
 * Testset 0; expected output = (-1) output from neural network = (-0.986415991978)
 * Testset 1; expected output = (1) output from neural network = (0.992121412998)
 * Testset 2; expected output = (1) output from neural network = (0.992469534962)
 * Testset 3; expected output = (-1) output from neural network = (-0.990224120384)
 * </code>
 * 
 * ...which indicates the network has learned the task. 
 *  
 * @author ir. E. Akerboom
 * @author {@link http://www.tremani.nl/ Tremani}, {@link http://maps.google.com/maps?f=q&hl=en&q=delft%2C+the+netherlands&ie=UTF8&t=k&om=1&ll=53.014783%2C4.921875&spn=36.882665%2C110.566406&z=4 Delft}, The Netherlands
 * @since feb 2007
 * @version 1.0
 * @license http://opensource.org/licenses/bsd-license.php BSD License
 */

class NeuralNetwork {

	/**#@+
	 * @access private
	 */
	var $nodecount = array ();
	var $nodevalue = array ();
	var $nodethreshold = array ();
	var $edgeweight = array ();
	var $learningrate = array (0.1);
	var $layercount = 0;
	var $previous_weightcorrection = array ();
	var $momentum = 0.8;
	var $is_verbose = true;

	var $trainInputs = array ();
	var $trainOutput = array ();
	var $trainDataID = array ();

	var $controlInputs = array ();
	var $controlOutput = array ();
	var $controlDataID = array ();

	var $weightsInitialized = false;

	var $epoch;
	var $error_trainingset;
	var $error_controlset;
	var $success;
	/**#@-*/

	/**
	 * Creates a neural network.
	 * 
	 * Example:
	 * <code>
	 * // create a network with 4 input nodes, 10 hidden nodes, and 4 output nodes
	 * $n = new NeuralNetwork(4, 10, 4);
	 * 
	 * // create a network with 4 input nodes, 1 hidden layer with 10 nodes, 
	 * // another hidden layer with 10 nodes, and 4 output nodes
	 * $n = new NeuralNetwork(4, 10, 10, 4); 
	 * 
	 * // alternative syntax
	 * $n = new NeuralNetwork(array(4, 10, 10, 4));
	 * </code>
	 * 
	 * @param array $nodecount The number of nodes in the consecutive layers.
	 */
	function NeuralNetwork($nodecount) {
		if (!is_array($nodecount)) {
			$nodecount = func_get_args();
		}
		$this->nodecount = $nodecount;

		// store the number of layers
		$this->layercount = count($this->nodecount);
	}

	/**
	 * Sets the learning rate between the different layers. 
	 *
	 * @param array $learningrate An array containing the learning rates [range 0.0 - 1.0]. 
	 * The size of this array is 'layercount - 1'. You might also provide a single number. If that is
	 * the case, then this will be the learning rate for the whole network.
	 */
	function setLearningRate($learningrate) {
		if (!is_array($learningrate)) {
			$learningrate = func_get_args();
		}

		$this->learningrate = $learningrate;
	}

	/**
	 * Gets the learning rate for a specific layer
	 * 
	 * @param int $layer The layer to obtain the learning rate for
	 * @return float The learning rate for that layer
	 */
	function getLearningRate($layer) {
		if (array_key_exists($layer, $this->learningrate)) {
			return $this->learningrate[$layer];
		}
		return $this->learningrate[0];
	}

	/**
	 * Sets the 'momentum' for the learning algorithm. The momentum should 
	 * accelerate the learning process and help avoid local minima.
	 * 
	 * @param float $momentum The momentum. Must be between 0.0 and 1.0; Usually between 0.5 and 0.9
	 */
	function setMomentum($momentum) {
		$this->momentum = $momentum;
	}

	/**
	 * Gets the momentum.
	 * 
	 * @return float The momentum
	 */
	function getMomentum() {
		return $this->momentum;
	}

	/**
	 * Calculate the output of the neural network for a given input vector
	 * 
	 * @param array $input The vector to calculate
	 * @return mixed The output of the network
	 */
	function calculate($input) {

		// put the input vector on the input nodes
		foreach ($input as $index => $value) {
			$this->nodevalue[0][$index] = $value;
		}

		// iterate the hidden layers
		for ($layer = 1; $layer < $this->layercount; $layer ++) {

			$prev_layer = $layer -1;

			// iterate each node in this layer
			for ($node = 0; $node < ($this->nodecount[$layer]); $node ++) {
				$node_value = 0.0;

				// each node in the previous layer has a connection to this node
				// on basis of this, calculate this node's value
				for ($prev_node = 0; $prev_node < ($this->nodecount[$prev_layer]); $prev_node ++) {
					$inputnode_value = $this->nodevalue[$prev_layer][$prev_node];
					$edge_weight = $this->edgeweight[$prev_layer][$prev_node][$node];

					$node_value = $node_value + ($inputnode_value * $edge_weight);
				}

				// apply the threshold
				$node_value = $node_value - $this->nodethreshold[$layer][$node];

				// apply the activation function
				$node_value = $this->activation($node_value);

				// remember the outcome
				$this->nodevalue[$layer][$node] = $node_value;
			}
		}

		// return the values of the last layer (the output layer)
		return $this->nodevalue[$this->layercount - 1];
	}

	/**
	 * Implements the standard (default) activation function for backpropagation networks, 
	 * the 'tanh' activation function.
	 * 
	 * @param float $value The preliminary output to apply this function to
	 * @return float The final output of the node
	 */
	function activation($value) {
		return tanh($value);
		// return (1.0 / (1.0 + exp(- $value)));
	}

	/**
	 * Implements the derivative of the activation function. By default, this is the 
	 * inverse of the 'tanh' activation function: 1.0 - tanh($value)*tanh($value);
	 * 
	 * @param float $value 'X'
	 * @return $float 
	 */
	function derivative_activation($value) {
		$tanh = tanh($value);
		return 1.0 - $tanh * $tanh;
		//return $value * (1.0 - $value);
	}

	/**
	 * Add a test vector and its output
	 * 
	 * @param array $input An input vector
	 * @param array $output The corresponding output
	 * @param int $id (optional) An identifier for this piece of data
	 */
	function addTestData($input, $output, $id = null) {
		$index = count($this->trainInputs);
		foreach ($input as $node => $value) {
			$this->trainInputs[$index][$node] = $value;
		}

		foreach ($output as $node => $value) {
			$this->trainOutput[$index][$node] = $value;
		}

		$this->trainDataID[$index] = $id;
	}

	/**
	 * Returns the identifiers of the data used to train the network (if available)
	 * 
	 * @return array An array of identifiers
	 */
	function getTestDataIDs() {
		return $this->trainDataID;
	}

	/**
	 * Add a set of control data to the network. 
	 * 
	 * This set of data is used to prevent 'overlearning' of the network. The 
	 * network will stop training if the results obtained for the control data 
	 * are worsening.
	 * 
	 * The data added as control data is not used for training.
	 * 
	 * @param array $input An input vector
	 * @param array $output The corresponding output
	* @param int $id (optional) An identifier for this piece of data
	 */
	function addControlData($input, $output, $id = null) {
		$index = count($this->controlInputs);
		foreach ($input as $node => $value) {
			$this->controlInputs[$index][$node] = $value;
		}

		foreach ($output as $node => $value) {
			$this->controlOutput[$index][$node] = $value;
		}

		$this->controlDataID[$index] = $id;
	}

	/**
	 * Returns the identifiers of the control data used during the training 
	 * of the network (if available)
	 * 
	 * @return array An array of identifiers
	 */
	function getControlDataIDs() {
		return $this->controlDataID;
	}

	/**
	 * Shows the current weights and thresholds
	 * 
	 * @param boolean $force Force the output, even if the network is {@link setVerbose() not verbose}. 
	 */
	function showWeights($force = false) {
		if ($this->isVerbose() || $force) {
			echo "<hr>";
			echo "<br />Weights: <pre>".serialize($this->edgeweight)."</pre>";
			echo "<br />Thresholds: <pre>".serialize($this->nodethreshold)."</pre>";
		}
	}

	/**
	 * Determines if the neural network displays status and error messages. By default, it does.
	 * 
	 * @param boolean $is_verbose 'true' if you want to display status and error messages, 'false' if you don't
	 */
	function setVerbose($is_verbose) {
		$this->is_verbose = $is_verbose;
	}

	/**
	 * Returns whether or not the network displays status and error messages.
	 * 
	 * @return boolean 'true' if status and error messages are displayed, 'false' otherwise
	 */
	function isVerbose() {
		return $this->is_verbose;
	}

	/**
	 * Loads a neural network from a file saved by the 'save()' function. Clears 
	 * the training and control data added so far.
	 * 
	 * @param string $filename The filename to load the network from
	 * @return boolean 'true' on success, 'false' otherwise
	 */
	function load($filename) {
		if (file_exists($filename)) {
			$data = parse_ini_file($filename);
			if (array_key_exists("edges", $data) && array_key_exists("thresholds", $data)) {
				// make sure all standard preparations performed
				$this->initWeights();

				// load data from file
				$this->edgeweight = unserialize($data['edges']);
				$this->nodethreshold = unserialize($data['thresholds']);

				$this->weightsInitialized = true;

				// load IDs of training and control set
				if (array_key_exists("training_data", $data) && array_key_exists("control_data", $data)) {

					// load the IDs
					$this->trainDataID = unserialize($data['training_data']);
					$this->controlDataID = unserialize($data['control_data']);

					// if we do not reset the training and control data here, then we end up
					// with a bunch of IDs that do not refer to the actual data we're training
					// the network with.
					$this->controlInputs = array ();
					$this->controlOutput = array ();

					$this->trainInputs = array ();
					$this->trainOutput = array ();
				}

				return true;
			}
		}

		return false;
	}

	/**
	 * Saves a neural network to a file
	 * 
	 * @param string $filename The filename to save the neural network to
	 * @return boolean 'true' on success, 'false' otherwise
	 */
	function save($filename) {
		$f = fopen($filename, "w");
		if ($f) {
			fwrite($f, "[weights]");
			fwrite($f, "\r\nedges = \"".serialize($this->edgeweight)."\"");
			fwrite($f, "\r\nthresholds = \"".serialize($this->nodethreshold)."\"");
			fwrite($f, "\r\n");
			fwrite($f, "[identifiers]");
			fwrite($f, "\r\ntraining_data = \"".serialize($this->trainDataID)."\"");
			fwrite($f, "\r\ncontrol_data = \"".serialize($this->controlDataID)."\"");
			fclose($f);

			return true;
		}

		return false;
	}

	/**
	 * Start the training process
	 * 
	 * @param int $maxEpochs The maximum number of epochs
	 * @param float $maxError The maximum squared error in the training data
	 * @return bool 'true' if the training was successful, 'false' otherwise
	 */
	function train($maxEpochs = 500, $maxError = 0.01) {

		if (!$this->weightsInitialized) {
			$this->initWeights();
		}

		if ($this->isVerbose()) {
			echo "<table>";
			echo "<tr><th>#</th><th>error(trainingdata)</th><th>error(controldata)</th><th>slope(error(controldata))</th></tr>";
		}

		$epoch = 0;
		$errorControlSet = array ();
		$avgErrorControlSet = array ();
		define('SAMPLE_COUNT', 10);
		do {
//                        echo "<tr><td colspan=10><b>epoch $epoch</b></td></tr>";
			for ($i = 0; $i < count($this->trainInputs); $i ++) {
				// select a training pattern at random
				$index = mt_rand(0, count($this->trainInputs) - 1);

				// determine the input, and the desired output
				$input = $this->trainInputs[$index];
				$desired_output = $this->trainOutput[$index];

				// calculate the actual output
				$output = $this->calculate($input);

//                              echo "<tr><td></td><td>Training set $i</td><td>input = (" . implode(", ", $input) . ")</td>";
//			 	echo "<td>desired = (" . implode(", ", $desired_output) . ")</td>";
//				echo "<td>output = (" . implode(", ", $output) .")</td></tr>";

				// change network weights
				$this->backpropagate($output, $desired_output);
			}

			// buy some time
			set_time_limit(300);

			//display the overall network error after each epoch
			$squaredError = $this->squaredErrorEpoch();
			if ($epoch % 2 == 0) {
				$squaredErrorControlSet = $this->squaredErrorControlSet();
				$errorControlSet[] = $squaredErrorControlSet;

				if (count($errorControlSet) > SAMPLE_COUNT) {
					$avgErrorControlSet[] = array_sum(array_slice($errorControlSet, -SAMPLE_COUNT)) / SAMPLE_COUNT;
				}

				list ($slope, $offset) = $this->fitLine($avgErrorControlSet);
				$controlset_msg = $squaredErrorControlSet;
			} else {
				$controlset_msg = "";
			}

			if ($this->isVerbose()) {
				echo "<tr><td><b>$epoch</b></td><td>$squaredError</td><td>$controlset_msg";
				echo "<script type='text/javascript'>window.scrollBy(0,100);</script>";
				echo "</td><td>$slope</td></tr>";
				echo "</td></tr>";

				flush();
				ob_flush();
			}

			// conditions for a 'successful' stop:
			// 1. the squared error is now lower than the provided maximum error
			$stop_1 = $squaredError <= $maxError || $squaredErrorControlSet <= $maxError;

			// conditions for an 'unsuccessful' stop
			// 1. the maximum number of epochs has been reached
			$stop_2 = $epoch ++ > $maxEpochs;

			// 2. the network's performance on the control data is getting worse
			$stop_3 = $slope > 0;

		} while (!$stop_1 && !$stop_2 && !$stop_3);

		$this->setEpoch($epoch);
		$this->setErrorTrainingSet($squaredError);
		$this->setErrorControlSet($squaredErrorControlSet);
		$this->setTrainingSuccessful($stop_1);

		if ($this->isVerbose()) {
			echo "</table>";
		}

		return $stop_1;
	}

	/**
	 * After training, this function is used to store the number of epochs the network 
	 * needed for training the network. An epoch is defined as the number of times 
	 * the complete trainingset is used for training.
	 * 
	 * @access private
	 * @param int $epoch 
	 */
	function setEpoch($epoch) {
		$this->epoch = $epoch;
	}

	/**
	 * Gets the number of epochs the network needed for training.
	 * 
	 * @access private
	 * @return int The number of epochs.
	 */
	function getEpoch() {
		return $this->epoch;
	}

	/**
	 * After training, this function is used to store the squared error between the
	 * desired output and the obtained output of the training data.
	 * 
	 * @access private
	 * @param float $error The squared error of the training data
	 */
	function setErrorTrainingSet($error) {
		$this->error_trainingset = $error;
	}

	/**
	 * Gets the squared error between the desired output and the obtained output of 
	 * the training data.
	 * 
	 * @access private
	 * @return float The squared error of the training data
	 */
	function getErrorTrainingSet() {
		return $this->error_trainingset;
	}

	/**
	 * After training, this function is used to store the squared error between the
	 * desired output and the obtained output of the control data.
	 * 
	 * @access private
	 * @param float $error The squared error of the control data
	 */
	function setErrorControlSet($error) {
		$this->error_controlset = $error;
	}

	/**
	 * Gets the squared error between the desired output and the obtained output of 
	 * the control data.
	 * 
	 * @access private
	 * @return float The squared error of the control data
	 */
	function getErrorControlSet() {
		return $this->error_controlset;
	}

	/**
	 * After training, this function is used to store whether or not the training
	 * was successful.
	 * 
	 * @access private
	 * @param bool $success 'true' if the training was successful, 'false' otherwise
	 */
	function setTrainingSuccessful($success) {
		$this->success = $success;
	}

	/**
	 * Determines if the training was successful.
	 * 
	 * @access private
	 * @return bool 'true' if the training was successful, 'false' otherwise
	 */
	function getTrainingSuccessful() {
		return $this->success;
	}

	/**
	 * Finds the least square fitting line for the given data. 
	 * 
	 * This function is used to determine if the network is overtraining itself. If 
	 * the line through the controlset's most recent squared errors is going 'up', 
	 * then it's time to stop training.
	 * 
	 * @access private
	 * @param array $data The points to fit a line to. The keys of this array represent 
	 *                    the 'x'-value of the point, the corresponding value is the 
	 *                    'y'-value of the point.
	 * @return array An array containing, respectively, the slope and the offset of the fitted line.
	 */
	function fitLine($data) {
		// based on 
		//    http://mathworld.wolfram.com/LeastSquaresFitting.html

		$n = count($data);

		if ($n > 1) {
			$sum_y = 0;
			$sum_x = 0;
			$sum_x2 = 0;
			$sum_xy = 0;
			foreach ($data as $x => $y) {
				$sum_x += $x;
				$sum_y += $y;
				$sum_x2 += $x * $x;
				$sum_xy += $x * $y;
			}

			// implementation of formula (12)
			$offset = ($sum_y * $sum_x2 - $sum_x * $sum_xy) / ($n * $sum_x2 - $sum_x * $sum_x);

			// implementation of formula (13)
			$slope = ($n * $sum_xy - $sum_x * $sum_y) / ($n * $sum_x2 - $sum_x * $sum_x);

			return array ($slope, $offset);
		} else {
			return array (0.0, 0.0);
		}
	}

	/**
	 * Gets a random weight between [-0.25 .. 0.25]. Used to initialize the network.
	 * 
	 * @return float A random weight
	 */
	function getRandomWeight($layer) {
		return ((mt_rand(0, 1000) / 1000) - 0.5) / 2;
	}

	/**
	 * Randomise the weights in the neural network
	 * 
	 * @access private
	 */
	function initWeights() {
		// assign a random value to each edge between the layers, and randomise each threshold
		//
		// 1. start at layer '1' (so skip the input layer)
		for ($layer = 1; $layer < $this->layercount; $layer ++) {

			$prev_layer = $layer -1;

			// 2. in this layer, walk each node
			for ($node = 0; $node < $this->nodecount[$layer]; $node ++) {

				// 3. randomise this node's threshold
				$this->nodethreshold[$layer][$node] = $this->getRandomWeight($layer);

				// 4. this node is connected to each node of the previous layer
				for ($prev_index = 0; $prev_index < $this->nodecount[$prev_layer]; $prev_index ++) {

					// 5. this is the 'edge' that needs to be reset / initialised
					$this->edgeweight[$prev_layer][$prev_index][$node] = $this->getRandomWeight($prev_layer);

					// 6. initialize the 'previous weightcorrection' at 0.0
					$this->previous_weightcorrection[$prev_layer][$prev_index] = 0.0;
				}
			}
		}
	}

	/**
	* Performs the backpropagation algorithm. This changes the weights and thresholds of the network.
	* 
	* @access private
	* @param array $output The output obtained by the network
	* @param array $desired_output The desired output
	*/
	function backpropagate($output, $desired_output) {

		$errorgradient = array ();
		$outputlayer = $this->layercount - 1;

		$momentum = $this->getMomentum();

		// Propagate the difference between output and desired output through the layers.
		for ($layer = $this->layercount - 1; $layer > 0; $layer --) {
			for ($node = 0; $node < $this->nodecount[$layer]; $node ++) {

				// step 1: determine errorgradient
				if ($layer == $outputlayer) {
					// for the output layer:
					// 1a. calculate error between desired output and actual output
					$error = $desired_output[$node] - $output[$node];

					// 1b. calculate errorgradient
					$errorgradient[$layer][$node] = $this->derivative_activation($output[$node]) * $error;
				} else {
					// for hidden layers:
					// 1a. sum the product of edgeweight and errorgradient of the 'next' layer
					$next_layer = $layer +1;

					$productsum = 0;
					for ($next_index = 0; $next_index < ($this->nodecount[$next_layer]); $next_index ++) {
						$_errorgradient = $errorgradient[$next_layer][$next_index];
						$_edgeweight = $this->edgeweight[$layer][$node][$next_index];

						$productsum = $productsum + $_errorgradient * $_edgeweight;
					}

					// 1b. calculate errorgradient
					$nodevalue = $this->nodevalue[$layer][$node];
					$errorgradient[$layer][$node] = $this->derivative_activation($nodevalue) * $productsum;
				}

				// step 2: use the errorgradient to determine a weight correction for each node
				$prev_layer = $layer -1;
				$learning_rate = $this->getLearningRate($prev_layer);

				for ($prev_index = 0; $prev_index < ($this->nodecount[$prev_layer]); $prev_index ++) {

					// 2a. obtain nodevalue, edgeweight and learning rate
					$nodevalue = $this->nodevalue[$prev_layer][$prev_index];
					$edgeweight = $this->edgeweight[$prev_layer][$prev_index][$node];

					// 2b. calculate weight correction
					$weight_correction = $learning_rate * $nodevalue * $errorgradient[$layer][$node];

					// 2c. retrieve previous weight correction
					$prev_weightcorrection = $this->previous_weightcorrection[$layer][$node];

					// 2d. combine those ('momentum learning') to a new weight
					$new_weight = $edgeweight + $weight_correction + $momentum * $prev_weightcorrection;

					// 2e. assign the new weight to this edge
					$this->edgeweight[$prev_layer][$prev_index][$node] = $new_weight;

					// 2f. remember this weightcorrection
					$this->previous_weightcorrection[$layer][$node] = $weight_correction;
				}

				// step 3: use the errorgradient to determine threshold correction
				$threshold_correction = $learning_rate * -1 * $errorgradient[$layer][$node];
				$new_threshold = $this->nodethreshold[$layer][$node] + $threshold_correction;

				$this->nodethreshold[$layer][$node] = $new_threshold;
			}
		}
	}

	/**
	 * Calculate the root-mean-squared error of the output, given the
	 * trainingdata.
	 * 
	 * @access private
	 * @return float The root-mean-squared error of the output
	 */
	function squaredErrorEpoch() {
		$RMSerror = 0.0;
		for ($i = 0; $i < count($this->trainInputs); $i ++) {
			$RMSerror += $this->squaredError($this->trainInputs[$i], $this->trainOutput[$i]);
		}
		$RMSerror = $RMSerror / count($this->trainInputs);

		return sqrt($RMSerror);
	}

	/**
	 * Calculate the root-mean-squared error of the output, given the
	 * controldata.
	 * 
	 * @access private
	 * @return float The root-mean-squared error of the output
	 */
	function squaredErrorControlSet() {

		if (count($this->controlInputs) == 0) {
			return 1.0;
		}

		$RMSerror = 0.0;
		for ($i = 0; $i < count($this->controlInputs); $i ++) {
			$RMSerror += $this->squaredError($this->controlInputs[$i], $this->controlOutput[$i]);
		}
		$RMSerror = $RMSerror / count($this->controlInputs);

		return sqrt($RMSerror);
	}

	/**
	 * Calculate the root-mean-squared error of the output, given the
	 * desired output.
	 * 
	 * @access private
	 * @param array $input The input to test
	 * @param array $desired_output The desired output
	 * @return float The root-mean-squared error of the output compared to the desired output
	 */
	function squaredError($input, $desired_output) {
		$output = $this->calculate($input);

		$RMSerror = 0.0;
		foreach ($output as $node => $value) {
			//calculate the error
			$error = $output[$node] - $desired_output[$node];

			$RMSerror = $RMSerror + ($error * $error);
		}

		return $RMSerror;
	}
}

