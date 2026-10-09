The test uses the local full dataset and released HDF5 checkpoint, so it is
opt-in rather than part of the fast unit-test suite:

    TEMNET_RUN_INTEGRATION=1 TEMNET_REQUIRE_GPU=1 \
    TEMNET_TEST_BACKBONE=resnet101 \
        python -m unittest tests.test_resnet101_pipeline -v

Set TEMNET_TEST_BACKBONE to "temnet", "resnet101v2", or
"inception_resnetv2" to exercise that backbone.
