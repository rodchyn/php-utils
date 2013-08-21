<?php

/*
 * This file is part of the CG library.
 *
 *    (C) 2011 Johannes M. Schmitt <schmittjoh@gmail.com>
 *
 */

spl_autoload_register(function($class)
{

    if (0 === strpos($class, 'Rodchyn\\Utils\\Tests\\')) {
        $path = __DIR__.'/../tests/'.strtr(preg_replace('/^Rodchyn\\\Utils\\\Tests\\\/', '', $class), '\\', '/').'.php';
        if (file_exists($path) && is_readable($path)) {
            require_once $path;

            return true;
        }
    }

    if (0 === strpos($class, 'Rodchyn\\Utils\\')) {
        $path = __DIR__.'/../lib/'.strtr($class, '\\', '/').'.php';
        echo $path;
        if (file_exists($path) && is_readable($path)) {
            require_once $path;

            return true;
        }
    }
});
