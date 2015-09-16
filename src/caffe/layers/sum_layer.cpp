#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/syncedmem.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

using std::min;
using std::max;
template <typename Dtype>
void SumLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>* >& bottom, vector<Blob<Dtype>*>* top)
{
	this->Reshape(bottom, top);
}


template <typename Dtype> 
void SumLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom, vector<Blob<Dtype>*>* top)
{
	(*top)[0]->Reshape(1,1,1,1);
}


template <typename Dtype>
void SumLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom, vector<Blob<Dtype>*>* top)
{
	Dtype sum_ = Dtype(0);
	for(int iter = 0; iter < (int)bottom.size(); iter++){
		const Dtype* bottom_data = bottom[iter]->cpu_data();
		for(int n = 0; n < bottom[iter]->num(); n++){
			for(int c = 0; c < bottom[iter]->channels(); c++){
				for(int h = 0; h < bottom[iter]->height(); h++){
					for(int w = 0; w < bottom[iter]->width(); w++){
						sum_ += bottom_data[((n*bottom[iter]->channels() + c)*bottom[iter]->height() + h)*bottom[iter]->width() + w];
					}
				}
			}
		}
	}

	(*top)[0]->mutable_cpu_data()[0] = sum_;
}


template <typename Dtype>
void SumLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
		vector<Blob<Dtype>*>* bottom)
{
	if(!propagate_down[0]){
		return;
	}

	const Dtype* top_diff = top[0]->cpu_diff();
	for(int iter = 0; iter < (int)(*bottom).size(); iter++ ){
		Dtype* bottom_diff = (*bottom)[iter]->mutable_cpu_diff();
		for(int n = 0; n < (*bottom)[iter]->num(); n++){
			for(int c = 0; c < (*bottom)[iter]->channels(); c++){
				for(int h = 0; h < (*bottom)[iter]->height(); h++){
					for(int w = 0; w < (*bottom)[iter]->width(); w++){
						bottom_diff[((n*(*bottom)[iter]->channels() + c)*(*bottom)[iter]->height() + h)*(*bottom)[iter]->width() + w]+= top_diff[0];
					}
				}
			}
		}
	}


}
#ifdef CPU_ONLY
STUB_GPU(SumLayer);
#endif

INSTANTIATE_CLASS(SumLayer);
}// namespace caffe



