<?php

namespace Rodchyn\Utils\Tests;

use Rodchyn\Utils\ExtendedArrayObject;

class ExtendedArrayObjectTest extends \PHPUnit_Framework_TestCase
{
	public function testArray()
	{
		$array = new ExtendedArrayObject(['1.0', '1.2']);
		$array->prepend('00');
        $array->insert('1.1', 2);

        $this->assertEquals('00', $array[0]);
        $this->assertEquals('1.0', $array[1]);
        $this->assertEquals('1.1', $array[2]);
        $this->assertEquals('1.2', $array[3]);
	}
}
