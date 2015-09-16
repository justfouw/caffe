#include <cstring>
#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/vision_layers.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

namespace caffe {

template <typename TypeParam>
class SumLayerTest : public MultiDeviceTest<TypeParam>	{
	typedef typename TypeParam::Dtype Dtype;
	protected:
		SumLayerTest()
			: blob_bottom_(new Blob<Dtype>(2, 10, 2, 3)),
			  blob_top_(new Blob<Dtype>())	{
			// fill the values
			FillerParameter filler_param;
			filler_param.set_value(0.5f);
			ConstantFiller<Dtype> filler(filler_param);		
			filler.Fill(this->blob_bottom_);
			blob_bottom_vec_.push_back(blob_bottom_);
			blob_top_vec_.push_back(blob_top_);
		}

		virtual ~SumLayerTest() {delete blob_bottom_; delete blob_top_;}
		Blob<Dtype>* const blob_bottom_;
		Blob<Dtype>* const blob_top_;
		vector<Blob<Dtype>*> blob_bottom_vec_;
		vector<Blob<Dtype>*> blob_top_vec_;		
};

TYPED_TEST_CASE(SumLayerTest, TestDtypesAndDevices);

TYPED_TEST(SumLayerTest, TestForward){
	typedef typename TypeParam::Dtype Dtype;
	LayerParameter layer_param;
	SumLayer<Dtype> layer(layer_param);
	layer.SetUp(this->blob_bottom_vec_, &(this->blob_top_vec_));
	layer.Forward(this->blob_bottom_vec_, &(this->blob_top_vec_));

	// test sum
	Dtype sum_ = Dtype(0);
	for(int iter = 0; iter < (int)this->blob_bottom_vec_.size(); iter++){
		for(int n = 0; n < this->blob_bottom_vec_[iter]->num(); n++ ){
			for(int c = 0; c < this->blob_bottom_vec_[iter]->channels(); c++){
				for(int h = 0; h < this->blob_bottom_vec_[iter]->height(); h++){
					for(int w = 0; w < this->blob_bottom_vec_[iter]->width(); w++){
						sum_ += this->blob_bottom_vec_[iter]->data_at(n, c, h, w);
					}
				}
			}
		}
	}

	EXPECT_GT(sum_, 59.999);
	EXPECT_LE(sum_, 60.001);
}



TYPED_TEST_CASE(SumLayerTest, TestDtypesAndDevices);
TYPED_TEST(SumLayerTest, TestGradient){
	typedef typename TypeParam:: Dtype Dtype;
	LayerParameter layer_param;
	SumLayer<Dtype> layer(layer_param);
	GradientChecker<Dtype> checker(1e-2, 1e-3);
	checker.CheckGradientExhaustive(&layer, &(this->blob_bottom_vec_), &(this->blob_top_vec_));
}
} // namespace caffe 
