<?php

namespace Rodchyn\Utils;

/**
 * Class ExtendedArrayObject
 * @package Rodchyn\Utils
 */
class ExtendedArrayObject extends ArrayObject {

    /**
     * @param $value mixed
     */
    public function prepend($value) {
        $array = (array)$this;
        array_unshift($array, $value);
        $this->exchangeArray($array);
    }

    /**
     * @param $insert mixed
     * @param $position int
     */
    public function insert($insert, $position) {
        $array = (array)$this;
        array_splice($array, $position, 0, $insert);
        $this->exchangeArray($array);
    }
}
