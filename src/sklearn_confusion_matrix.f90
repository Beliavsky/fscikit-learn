! sklearn 1.6.1 metrics/_classification.py confusion_matrix (BSD-3-Clause).
! Explicit typed adapters and integer/real result integration are MIT.
module sklearn_confusion_matrix
 use sklearn_common,only:dp,require
 use sklearn_categorical_scores,only:confusion_values
 implicit none
 private
 public::confusion_matrix
 interface confusion_matrix
  module procedure counts_iv_iv
  module procedure counts_iv_im
  module procedure counts_iv_rv
  module procedure counts_iv_rm
  module procedure counts_iv_lv
  module procedure counts_iv_lm
  module procedure counts_im_iv
  module procedure counts_im_im
  module procedure counts_im_rv
  module procedure counts_im_rm
  module procedure counts_im_lv
  module procedure counts_im_lm
  module procedure counts_rv_iv
  module procedure counts_rv_im
  module procedure counts_rv_rv
  module procedure counts_rv_rm
  module procedure counts_rv_lv
  module procedure counts_rv_lm
  module procedure counts_rm_iv
  module procedure counts_rm_im
  module procedure counts_rm_rv
  module procedure counts_rm_rm
  module procedure counts_rm_lv
  module procedure counts_rm_lm
  module procedure counts_lv_iv
  module procedure counts_lv_im
  module procedure counts_lv_rv
  module procedure counts_lv_rm
  module procedure counts_lv_lv
  module procedure counts_lv_lm
  module procedure counts_lm_iv
  module procedure counts_lm_im
  module procedure counts_lm_rv
  module procedure counts_lm_rm
  module procedure counts_lm_lv
  module procedure counts_lm_lm
  module procedure weighted_iv_iv
  module procedure weighted_iv_im
  module procedure weighted_iv_rv
  module procedure weighted_iv_rm
  module procedure weighted_iv_lv
  module procedure weighted_iv_lm
  module procedure weighted_im_iv
  module procedure weighted_im_im
  module procedure weighted_im_rv
  module procedure weighted_im_rm
  module procedure weighted_im_lv
  module procedure weighted_im_lm
  module procedure weighted_rv_iv
  module procedure weighted_rv_im
  module procedure weighted_rv_rv
  module procedure weighted_rv_rm
  module procedure weighted_rv_lv
  module procedure weighted_rv_lm
  module procedure weighted_rm_iv
  module procedure weighted_rm_im
  module procedure weighted_rm_rv
  module procedure weighted_rm_rm
  module procedure weighted_rm_lv
  module procedure weighted_rm_lm
  module procedure weighted_lv_iv
  module procedure weighted_lv_im
  module procedure weighted_lv_rv
  module procedure weighted_lv_rm
  module procedure weighted_lv_lv
  module procedure weighted_lv_lm
  module procedure weighted_lm_iv
  module procedure weighted_lm_im
  module procedure weighted_lm_rv
  module procedure weighted_lm_rm
  module procedure weighted_lm_lv
  module procedure weighted_lm_lm
  module procedure normalized_iv_iv
  module procedure normalized_iv_im
  module procedure normalized_iv_rv
  module procedure normalized_iv_rm
  module procedure normalized_iv_lv
  module procedure normalized_iv_lm
  module procedure normalized_im_iv
  module procedure normalized_im_im
  module procedure normalized_im_rv
  module procedure normalized_im_rm
  module procedure normalized_im_lv
  module procedure normalized_im_lm
  module procedure normalized_rv_iv
  module procedure normalized_rv_im
  module procedure normalized_rv_rv
  module procedure normalized_rv_rm
  module procedure normalized_rv_lv
  module procedure normalized_rv_lm
  module procedure normalized_rm_iv
  module procedure normalized_rm_im
  module procedure normalized_rm_rv
  module procedure normalized_rm_rm
  module procedure normalized_rm_lv
  module procedure normalized_rm_lm
  module procedure normalized_lv_iv
  module procedure normalized_lv_im
  module procedure normalized_lv_rv
  module procedure normalized_lv_rm
  module procedure normalized_lv_lv
  module procedure normalized_lv_lm
  module procedure normalized_lm_iv
  module procedure normalized_lm_im
  module procedure normalized_lm_rv
  module procedure normalized_lm_rm
  module procedure normalized_lm_lv
  module procedure normalized_lm_lm
 end interface
contains
 function target_i(x) result(v)
  integer,intent(in)::x(..)
  real(dp),allocatable::v(:)
  select rank(x)
  rank(1);v=real(x,dp)
  rank(2)
   call require(size(x,1)>0.and.size(x,2)==1,'confusion matrices require nonempty single-column targets')
   v=real(x(:,1),dp)
  end select
 end function
 function target_r(x) result(v)
  real(dp),intent(in)::x(..)
  real(dp),allocatable::v(:)
  select rank(x)
  rank(1);v=x
  rank(2)
   call require(size(x,1)>0.and.size(x,2)==1,'confusion matrices require nonempty single-column targets')
   v=x(:,1)
  end select
 end function
 function target_l(x) result(v)
  logical,intent(in)::x(..)
  real(dp),allocatable::v(:)
  select rank(x)
  rank(1);v=merge(1._dp,0._dp,x)
  rank(2)
   call require(size(x,1)>0.and.size(x,2)==1,'confusion matrices require nonempty single-column targets')
   v=merge(1._dp,0._dp,x(:,1))
  end select
 end function
 function counts_iv_iv(y,p,labels) result(cm)
  integer,intent(in)::y(:)
  integer,intent(in)::p(:)
  class(*),optional,intent(in)::labels(:)
  integer,allocatable::cm(:,:)
  cm=int(confusion_values(target_i(y),target_i(p),labels))
 end function
 function counts_iv_im(y,p,labels) result(cm)
  integer,intent(in)::y(:)
  integer,intent(in)::p(:,:)
  class(*),optional,intent(in)::labels(:)
  integer,allocatable::cm(:,:)
  cm=int(confusion_values(target_i(y),target_i(p),labels))
 end function
 function counts_iv_rv(y,p,labels) result(cm)
  integer,intent(in)::y(:)
  real(dp),intent(in)::p(:)
  class(*),optional,intent(in)::labels(:)
  integer,allocatable::cm(:,:)
  cm=int(confusion_values(target_i(y),target_r(p),labels))
 end function
 function counts_iv_rm(y,p,labels) result(cm)
  integer,intent(in)::y(:)
  real(dp),intent(in)::p(:,:)
  class(*),optional,intent(in)::labels(:)
  integer,allocatable::cm(:,:)
  cm=int(confusion_values(target_i(y),target_r(p),labels))
 end function
 function counts_iv_lv(y,p,labels) result(cm)
  integer,intent(in)::y(:)
  logical,intent(in)::p(:)
  class(*),optional,intent(in)::labels(:)
  integer,allocatable::cm(:,:)
  cm=int(confusion_values(target_i(y),target_l(p),labels))
 end function
 function counts_iv_lm(y,p,labels) result(cm)
  integer,intent(in)::y(:)
  logical,intent(in)::p(:,:)
  class(*),optional,intent(in)::labels(:)
  integer,allocatable::cm(:,:)
  cm=int(confusion_values(target_i(y),target_l(p),labels))
 end function
 function counts_im_iv(y,p,labels) result(cm)
  integer,intent(in)::y(:,:)
  integer,intent(in)::p(:)
  class(*),optional,intent(in)::labels(:)
  integer,allocatable::cm(:,:)
  cm=int(confusion_values(target_i(y),target_i(p),labels))
 end function
 function counts_im_im(y,p,labels) result(cm)
  integer,intent(in)::y(:,:)
  integer,intent(in)::p(:,:)
  class(*),optional,intent(in)::labels(:)
  integer,allocatable::cm(:,:)
  cm=int(confusion_values(target_i(y),target_i(p),labels))
 end function
 function counts_im_rv(y,p,labels) result(cm)
  integer,intent(in)::y(:,:)
  real(dp),intent(in)::p(:)
  class(*),optional,intent(in)::labels(:)
  integer,allocatable::cm(:,:)
  cm=int(confusion_values(target_i(y),target_r(p),labels))
 end function
 function counts_im_rm(y,p,labels) result(cm)
  integer,intent(in)::y(:,:)
  real(dp),intent(in)::p(:,:)
  class(*),optional,intent(in)::labels(:)
  integer,allocatable::cm(:,:)
  cm=int(confusion_values(target_i(y),target_r(p),labels))
 end function
 function counts_im_lv(y,p,labels) result(cm)
  integer,intent(in)::y(:,:)
  logical,intent(in)::p(:)
  class(*),optional,intent(in)::labels(:)
  integer,allocatable::cm(:,:)
  cm=int(confusion_values(target_i(y),target_l(p),labels))
 end function
 function counts_im_lm(y,p,labels) result(cm)
  integer,intent(in)::y(:,:)
  logical,intent(in)::p(:,:)
  class(*),optional,intent(in)::labels(:)
  integer,allocatable::cm(:,:)
  cm=int(confusion_values(target_i(y),target_l(p),labels))
 end function
 function counts_rv_iv(y,p,labels) result(cm)
  real(dp),intent(in)::y(:)
  integer,intent(in)::p(:)
  class(*),optional,intent(in)::labels(:)
  integer,allocatable::cm(:,:)
  cm=int(confusion_values(target_r(y),target_i(p),labels))
 end function
 function counts_rv_im(y,p,labels) result(cm)
  real(dp),intent(in)::y(:)
  integer,intent(in)::p(:,:)
  class(*),optional,intent(in)::labels(:)
  integer,allocatable::cm(:,:)
  cm=int(confusion_values(target_r(y),target_i(p),labels))
 end function
 function counts_rv_rv(y,p,labels) result(cm)
  real(dp),intent(in)::y(:)
  real(dp),intent(in)::p(:)
  class(*),optional,intent(in)::labels(:)
  integer,allocatable::cm(:,:)
  cm=int(confusion_values(target_r(y),target_r(p),labels))
 end function
 function counts_rv_rm(y,p,labels) result(cm)
  real(dp),intent(in)::y(:)
  real(dp),intent(in)::p(:,:)
  class(*),optional,intent(in)::labels(:)
  integer,allocatable::cm(:,:)
  cm=int(confusion_values(target_r(y),target_r(p),labels))
 end function
 function counts_rv_lv(y,p,labels) result(cm)
  real(dp),intent(in)::y(:)
  logical,intent(in)::p(:)
  class(*),optional,intent(in)::labels(:)
  integer,allocatable::cm(:,:)
  cm=int(confusion_values(target_r(y),target_l(p),labels))
 end function
 function counts_rv_lm(y,p,labels) result(cm)
  real(dp),intent(in)::y(:)
  logical,intent(in)::p(:,:)
  class(*),optional,intent(in)::labels(:)
  integer,allocatable::cm(:,:)
  cm=int(confusion_values(target_r(y),target_l(p),labels))
 end function
 function counts_rm_iv(y,p,labels) result(cm)
  real(dp),intent(in)::y(:,:)
  integer,intent(in)::p(:)
  class(*),optional,intent(in)::labels(:)
  integer,allocatable::cm(:,:)
  cm=int(confusion_values(target_r(y),target_i(p),labels))
 end function
 function counts_rm_im(y,p,labels) result(cm)
  real(dp),intent(in)::y(:,:)
  integer,intent(in)::p(:,:)
  class(*),optional,intent(in)::labels(:)
  integer,allocatable::cm(:,:)
  cm=int(confusion_values(target_r(y),target_i(p),labels))
 end function
 function counts_rm_rv(y,p,labels) result(cm)
  real(dp),intent(in)::y(:,:)
  real(dp),intent(in)::p(:)
  class(*),optional,intent(in)::labels(:)
  integer,allocatable::cm(:,:)
  cm=int(confusion_values(target_r(y),target_r(p),labels))
 end function
 function counts_rm_rm(y,p,labels) result(cm)
  real(dp),intent(in)::y(:,:)
  real(dp),intent(in)::p(:,:)
  class(*),optional,intent(in)::labels(:)
  integer,allocatable::cm(:,:)
  cm=int(confusion_values(target_r(y),target_r(p),labels))
 end function
 function counts_rm_lv(y,p,labels) result(cm)
  real(dp),intent(in)::y(:,:)
  logical,intent(in)::p(:)
  class(*),optional,intent(in)::labels(:)
  integer,allocatable::cm(:,:)
  cm=int(confusion_values(target_r(y),target_l(p),labels))
 end function
 function counts_rm_lm(y,p,labels) result(cm)
  real(dp),intent(in)::y(:,:)
  logical,intent(in)::p(:,:)
  class(*),optional,intent(in)::labels(:)
  integer,allocatable::cm(:,:)
  cm=int(confusion_values(target_r(y),target_l(p),labels))
 end function
 function counts_lv_iv(y,p,labels) result(cm)
  logical,intent(in)::y(:)
  integer,intent(in)::p(:)
  class(*),optional,intent(in)::labels(:)
  integer,allocatable::cm(:,:)
  cm=int(confusion_values(target_l(y),target_i(p),labels))
 end function
 function counts_lv_im(y,p,labels) result(cm)
  logical,intent(in)::y(:)
  integer,intent(in)::p(:,:)
  class(*),optional,intent(in)::labels(:)
  integer,allocatable::cm(:,:)
  cm=int(confusion_values(target_l(y),target_i(p),labels))
 end function
 function counts_lv_rv(y,p,labels) result(cm)
  logical,intent(in)::y(:)
  real(dp),intent(in)::p(:)
  class(*),optional,intent(in)::labels(:)
  integer,allocatable::cm(:,:)
  cm=int(confusion_values(target_l(y),target_r(p),labels))
 end function
 function counts_lv_rm(y,p,labels) result(cm)
  logical,intent(in)::y(:)
  real(dp),intent(in)::p(:,:)
  class(*),optional,intent(in)::labels(:)
  integer,allocatable::cm(:,:)
  cm=int(confusion_values(target_l(y),target_r(p),labels))
 end function
 function counts_lv_lv(y,p,labels) result(cm)
  logical,intent(in)::y(:)
  logical,intent(in)::p(:)
  class(*),optional,intent(in)::labels(:)
  integer,allocatable::cm(:,:)
  cm=int(confusion_values(target_l(y),target_l(p),labels))
 end function
 function counts_lv_lm(y,p,labels) result(cm)
  logical,intent(in)::y(:)
  logical,intent(in)::p(:,:)
  class(*),optional,intent(in)::labels(:)
  integer,allocatable::cm(:,:)
  cm=int(confusion_values(target_l(y),target_l(p),labels))
 end function
 function counts_lm_iv(y,p,labels) result(cm)
  logical,intent(in)::y(:,:)
  integer,intent(in)::p(:)
  class(*),optional,intent(in)::labels(:)
  integer,allocatable::cm(:,:)
  cm=int(confusion_values(target_l(y),target_i(p),labels))
 end function
 function counts_lm_im(y,p,labels) result(cm)
  logical,intent(in)::y(:,:)
  integer,intent(in)::p(:,:)
  class(*),optional,intent(in)::labels(:)
  integer,allocatable::cm(:,:)
  cm=int(confusion_values(target_l(y),target_i(p),labels))
 end function
 function counts_lm_rv(y,p,labels) result(cm)
  logical,intent(in)::y(:,:)
  real(dp),intent(in)::p(:)
  class(*),optional,intent(in)::labels(:)
  integer,allocatable::cm(:,:)
  cm=int(confusion_values(target_l(y),target_r(p),labels))
 end function
 function counts_lm_rm(y,p,labels) result(cm)
  logical,intent(in)::y(:,:)
  real(dp),intent(in)::p(:,:)
  class(*),optional,intent(in)::labels(:)
  integer,allocatable::cm(:,:)
  cm=int(confusion_values(target_l(y),target_r(p),labels))
 end function
 function counts_lm_lv(y,p,labels) result(cm)
  logical,intent(in)::y(:,:)
  logical,intent(in)::p(:)
  class(*),optional,intent(in)::labels(:)
  integer,allocatable::cm(:,:)
  cm=int(confusion_values(target_l(y),target_l(p),labels))
 end function
 function counts_lm_lm(y,p,labels) result(cm)
  logical,intent(in)::y(:,:)
  logical,intent(in)::p(:,:)
  class(*),optional,intent(in)::labels(:)
  integer,allocatable::cm(:,:)
  cm=int(confusion_values(target_l(y),target_l(p),labels))
 end function
 function weighted_iv_iv(y,p,labels,sample_weight,normalize) result(cm)
  integer,intent(in)::y(:)
  integer,intent(in)::p(:)
  class(*),optional,intent(in)::labels(:)
  real(dp),intent(in)::sample_weight(:)
  character(*),optional,intent(in)::normalize
  real(dp),allocatable::cm(:,:)
  cm=confusion_values(target_i(y),target_i(p),labels,sample_weight,normalize)
 end function
 function weighted_iv_im(y,p,labels,sample_weight,normalize) result(cm)
  integer,intent(in)::y(:)
  integer,intent(in)::p(:,:)
  class(*),optional,intent(in)::labels(:)
  real(dp),intent(in)::sample_weight(:)
  character(*),optional,intent(in)::normalize
  real(dp),allocatable::cm(:,:)
  cm=confusion_values(target_i(y),target_i(p),labels,sample_weight,normalize)
 end function
 function weighted_iv_rv(y,p,labels,sample_weight,normalize) result(cm)
  integer,intent(in)::y(:)
  real(dp),intent(in)::p(:)
  class(*),optional,intent(in)::labels(:)
  real(dp),intent(in)::sample_weight(:)
  character(*),optional,intent(in)::normalize
  real(dp),allocatable::cm(:,:)
  cm=confusion_values(target_i(y),target_r(p),labels,sample_weight,normalize)
 end function
 function weighted_iv_rm(y,p,labels,sample_weight,normalize) result(cm)
  integer,intent(in)::y(:)
  real(dp),intent(in)::p(:,:)
  class(*),optional,intent(in)::labels(:)
  real(dp),intent(in)::sample_weight(:)
  character(*),optional,intent(in)::normalize
  real(dp),allocatable::cm(:,:)
  cm=confusion_values(target_i(y),target_r(p),labels,sample_weight,normalize)
 end function
 function weighted_iv_lv(y,p,labels,sample_weight,normalize) result(cm)
  integer,intent(in)::y(:)
  logical,intent(in)::p(:)
  class(*),optional,intent(in)::labels(:)
  real(dp),intent(in)::sample_weight(:)
  character(*),optional,intent(in)::normalize
  real(dp),allocatable::cm(:,:)
  cm=confusion_values(target_i(y),target_l(p),labels,sample_weight,normalize)
 end function
 function weighted_iv_lm(y,p,labels,sample_weight,normalize) result(cm)
  integer,intent(in)::y(:)
  logical,intent(in)::p(:,:)
  class(*),optional,intent(in)::labels(:)
  real(dp),intent(in)::sample_weight(:)
  character(*),optional,intent(in)::normalize
  real(dp),allocatable::cm(:,:)
  cm=confusion_values(target_i(y),target_l(p),labels,sample_weight,normalize)
 end function
 function weighted_im_iv(y,p,labels,sample_weight,normalize) result(cm)
  integer,intent(in)::y(:,:)
  integer,intent(in)::p(:)
  class(*),optional,intent(in)::labels(:)
  real(dp),intent(in)::sample_weight(:)
  character(*),optional,intent(in)::normalize
  real(dp),allocatable::cm(:,:)
  cm=confusion_values(target_i(y),target_i(p),labels,sample_weight,normalize)
 end function
 function weighted_im_im(y,p,labels,sample_weight,normalize) result(cm)
  integer,intent(in)::y(:,:)
  integer,intent(in)::p(:,:)
  class(*),optional,intent(in)::labels(:)
  real(dp),intent(in)::sample_weight(:)
  character(*),optional,intent(in)::normalize
  real(dp),allocatable::cm(:,:)
  cm=confusion_values(target_i(y),target_i(p),labels,sample_weight,normalize)
 end function
 function weighted_im_rv(y,p,labels,sample_weight,normalize) result(cm)
  integer,intent(in)::y(:,:)
  real(dp),intent(in)::p(:)
  class(*),optional,intent(in)::labels(:)
  real(dp),intent(in)::sample_weight(:)
  character(*),optional,intent(in)::normalize
  real(dp),allocatable::cm(:,:)
  cm=confusion_values(target_i(y),target_r(p),labels,sample_weight,normalize)
 end function
 function weighted_im_rm(y,p,labels,sample_weight,normalize) result(cm)
  integer,intent(in)::y(:,:)
  real(dp),intent(in)::p(:,:)
  class(*),optional,intent(in)::labels(:)
  real(dp),intent(in)::sample_weight(:)
  character(*),optional,intent(in)::normalize
  real(dp),allocatable::cm(:,:)
  cm=confusion_values(target_i(y),target_r(p),labels,sample_weight,normalize)
 end function
 function weighted_im_lv(y,p,labels,sample_weight,normalize) result(cm)
  integer,intent(in)::y(:,:)
  logical,intent(in)::p(:)
  class(*),optional,intent(in)::labels(:)
  real(dp),intent(in)::sample_weight(:)
  character(*),optional,intent(in)::normalize
  real(dp),allocatable::cm(:,:)
  cm=confusion_values(target_i(y),target_l(p),labels,sample_weight,normalize)
 end function
 function weighted_im_lm(y,p,labels,sample_weight,normalize) result(cm)
  integer,intent(in)::y(:,:)
  logical,intent(in)::p(:,:)
  class(*),optional,intent(in)::labels(:)
  real(dp),intent(in)::sample_weight(:)
  character(*),optional,intent(in)::normalize
  real(dp),allocatable::cm(:,:)
  cm=confusion_values(target_i(y),target_l(p),labels,sample_weight,normalize)
 end function
 function weighted_rv_iv(y,p,labels,sample_weight,normalize) result(cm)
  real(dp),intent(in)::y(:)
  integer,intent(in)::p(:)
  class(*),optional,intent(in)::labels(:)
  real(dp),intent(in)::sample_weight(:)
  character(*),optional,intent(in)::normalize
  real(dp),allocatable::cm(:,:)
  cm=confusion_values(target_r(y),target_i(p),labels,sample_weight,normalize)
 end function
 function weighted_rv_im(y,p,labels,sample_weight,normalize) result(cm)
  real(dp),intent(in)::y(:)
  integer,intent(in)::p(:,:)
  class(*),optional,intent(in)::labels(:)
  real(dp),intent(in)::sample_weight(:)
  character(*),optional,intent(in)::normalize
  real(dp),allocatable::cm(:,:)
  cm=confusion_values(target_r(y),target_i(p),labels,sample_weight,normalize)
 end function
 function weighted_rv_rv(y,p,labels,sample_weight,normalize) result(cm)
  real(dp),intent(in)::y(:)
  real(dp),intent(in)::p(:)
  class(*),optional,intent(in)::labels(:)
  real(dp),intent(in)::sample_weight(:)
  character(*),optional,intent(in)::normalize
  real(dp),allocatable::cm(:,:)
  cm=confusion_values(target_r(y),target_r(p),labels,sample_weight,normalize)
 end function
 function weighted_rv_rm(y,p,labels,sample_weight,normalize) result(cm)
  real(dp),intent(in)::y(:)
  real(dp),intent(in)::p(:,:)
  class(*),optional,intent(in)::labels(:)
  real(dp),intent(in)::sample_weight(:)
  character(*),optional,intent(in)::normalize
  real(dp),allocatable::cm(:,:)
  cm=confusion_values(target_r(y),target_r(p),labels,sample_weight,normalize)
 end function
 function weighted_rv_lv(y,p,labels,sample_weight,normalize) result(cm)
  real(dp),intent(in)::y(:)
  logical,intent(in)::p(:)
  class(*),optional,intent(in)::labels(:)
  real(dp),intent(in)::sample_weight(:)
  character(*),optional,intent(in)::normalize
  real(dp),allocatable::cm(:,:)
  cm=confusion_values(target_r(y),target_l(p),labels,sample_weight,normalize)
 end function
 function weighted_rv_lm(y,p,labels,sample_weight,normalize) result(cm)
  real(dp),intent(in)::y(:)
  logical,intent(in)::p(:,:)
  class(*),optional,intent(in)::labels(:)
  real(dp),intent(in)::sample_weight(:)
  character(*),optional,intent(in)::normalize
  real(dp),allocatable::cm(:,:)
  cm=confusion_values(target_r(y),target_l(p),labels,sample_weight,normalize)
 end function
 function weighted_rm_iv(y,p,labels,sample_weight,normalize) result(cm)
  real(dp),intent(in)::y(:,:)
  integer,intent(in)::p(:)
  class(*),optional,intent(in)::labels(:)
  real(dp),intent(in)::sample_weight(:)
  character(*),optional,intent(in)::normalize
  real(dp),allocatable::cm(:,:)
  cm=confusion_values(target_r(y),target_i(p),labels,sample_weight,normalize)
 end function
 function weighted_rm_im(y,p,labels,sample_weight,normalize) result(cm)
  real(dp),intent(in)::y(:,:)
  integer,intent(in)::p(:,:)
  class(*),optional,intent(in)::labels(:)
  real(dp),intent(in)::sample_weight(:)
  character(*),optional,intent(in)::normalize
  real(dp),allocatable::cm(:,:)
  cm=confusion_values(target_r(y),target_i(p),labels,sample_weight,normalize)
 end function
 function weighted_rm_rv(y,p,labels,sample_weight,normalize) result(cm)
  real(dp),intent(in)::y(:,:)
  real(dp),intent(in)::p(:)
  class(*),optional,intent(in)::labels(:)
  real(dp),intent(in)::sample_weight(:)
  character(*),optional,intent(in)::normalize
  real(dp),allocatable::cm(:,:)
  cm=confusion_values(target_r(y),target_r(p),labels,sample_weight,normalize)
 end function
 function weighted_rm_rm(y,p,labels,sample_weight,normalize) result(cm)
  real(dp),intent(in)::y(:,:)
  real(dp),intent(in)::p(:,:)
  class(*),optional,intent(in)::labels(:)
  real(dp),intent(in)::sample_weight(:)
  character(*),optional,intent(in)::normalize
  real(dp),allocatable::cm(:,:)
  cm=confusion_values(target_r(y),target_r(p),labels,sample_weight,normalize)
 end function
 function weighted_rm_lv(y,p,labels,sample_weight,normalize) result(cm)
  real(dp),intent(in)::y(:,:)
  logical,intent(in)::p(:)
  class(*),optional,intent(in)::labels(:)
  real(dp),intent(in)::sample_weight(:)
  character(*),optional,intent(in)::normalize
  real(dp),allocatable::cm(:,:)
  cm=confusion_values(target_r(y),target_l(p),labels,sample_weight,normalize)
 end function
 function weighted_rm_lm(y,p,labels,sample_weight,normalize) result(cm)
  real(dp),intent(in)::y(:,:)
  logical,intent(in)::p(:,:)
  class(*),optional,intent(in)::labels(:)
  real(dp),intent(in)::sample_weight(:)
  character(*),optional,intent(in)::normalize
  real(dp),allocatable::cm(:,:)
  cm=confusion_values(target_r(y),target_l(p),labels,sample_weight,normalize)
 end function
 function weighted_lv_iv(y,p,labels,sample_weight,normalize) result(cm)
  logical,intent(in)::y(:)
  integer,intent(in)::p(:)
  class(*),optional,intent(in)::labels(:)
  real(dp),intent(in)::sample_weight(:)
  character(*),optional,intent(in)::normalize
  real(dp),allocatable::cm(:,:)
  cm=confusion_values(target_l(y),target_i(p),labels,sample_weight,normalize)
 end function
 function weighted_lv_im(y,p,labels,sample_weight,normalize) result(cm)
  logical,intent(in)::y(:)
  integer,intent(in)::p(:,:)
  class(*),optional,intent(in)::labels(:)
  real(dp),intent(in)::sample_weight(:)
  character(*),optional,intent(in)::normalize
  real(dp),allocatable::cm(:,:)
  cm=confusion_values(target_l(y),target_i(p),labels,sample_weight,normalize)
 end function
 function weighted_lv_rv(y,p,labels,sample_weight,normalize) result(cm)
  logical,intent(in)::y(:)
  real(dp),intent(in)::p(:)
  class(*),optional,intent(in)::labels(:)
  real(dp),intent(in)::sample_weight(:)
  character(*),optional,intent(in)::normalize
  real(dp),allocatable::cm(:,:)
  cm=confusion_values(target_l(y),target_r(p),labels,sample_weight,normalize)
 end function
 function weighted_lv_rm(y,p,labels,sample_weight,normalize) result(cm)
  logical,intent(in)::y(:)
  real(dp),intent(in)::p(:,:)
  class(*),optional,intent(in)::labels(:)
  real(dp),intent(in)::sample_weight(:)
  character(*),optional,intent(in)::normalize
  real(dp),allocatable::cm(:,:)
  cm=confusion_values(target_l(y),target_r(p),labels,sample_weight,normalize)
 end function
 function weighted_lv_lv(y,p,labels,sample_weight,normalize) result(cm)
  logical,intent(in)::y(:)
  logical,intent(in)::p(:)
  class(*),optional,intent(in)::labels(:)
  real(dp),intent(in)::sample_weight(:)
  character(*),optional,intent(in)::normalize
  real(dp),allocatable::cm(:,:)
  cm=confusion_values(target_l(y),target_l(p),labels,sample_weight,normalize)
 end function
 function weighted_lv_lm(y,p,labels,sample_weight,normalize) result(cm)
  logical,intent(in)::y(:)
  logical,intent(in)::p(:,:)
  class(*),optional,intent(in)::labels(:)
  real(dp),intent(in)::sample_weight(:)
  character(*),optional,intent(in)::normalize
  real(dp),allocatable::cm(:,:)
  cm=confusion_values(target_l(y),target_l(p),labels,sample_weight,normalize)
 end function
 function weighted_lm_iv(y,p,labels,sample_weight,normalize) result(cm)
  logical,intent(in)::y(:,:)
  integer,intent(in)::p(:)
  class(*),optional,intent(in)::labels(:)
  real(dp),intent(in)::sample_weight(:)
  character(*),optional,intent(in)::normalize
  real(dp),allocatable::cm(:,:)
  cm=confusion_values(target_l(y),target_i(p),labels,sample_weight,normalize)
 end function
 function weighted_lm_im(y,p,labels,sample_weight,normalize) result(cm)
  logical,intent(in)::y(:,:)
  integer,intent(in)::p(:,:)
  class(*),optional,intent(in)::labels(:)
  real(dp),intent(in)::sample_weight(:)
  character(*),optional,intent(in)::normalize
  real(dp),allocatable::cm(:,:)
  cm=confusion_values(target_l(y),target_i(p),labels,sample_weight,normalize)
 end function
 function weighted_lm_rv(y,p,labels,sample_weight,normalize) result(cm)
  logical,intent(in)::y(:,:)
  real(dp),intent(in)::p(:)
  class(*),optional,intent(in)::labels(:)
  real(dp),intent(in)::sample_weight(:)
  character(*),optional,intent(in)::normalize
  real(dp),allocatable::cm(:,:)
  cm=confusion_values(target_l(y),target_r(p),labels,sample_weight,normalize)
 end function
 function weighted_lm_rm(y,p,labels,sample_weight,normalize) result(cm)
  logical,intent(in)::y(:,:)
  real(dp),intent(in)::p(:,:)
  class(*),optional,intent(in)::labels(:)
  real(dp),intent(in)::sample_weight(:)
  character(*),optional,intent(in)::normalize
  real(dp),allocatable::cm(:,:)
  cm=confusion_values(target_l(y),target_r(p),labels,sample_weight,normalize)
 end function
 function weighted_lm_lv(y,p,labels,sample_weight,normalize) result(cm)
  logical,intent(in)::y(:,:)
  logical,intent(in)::p(:)
  class(*),optional,intent(in)::labels(:)
  real(dp),intent(in)::sample_weight(:)
  character(*),optional,intent(in)::normalize
  real(dp),allocatable::cm(:,:)
  cm=confusion_values(target_l(y),target_l(p),labels,sample_weight,normalize)
 end function
 function weighted_lm_lm(y,p,labels,sample_weight,normalize) result(cm)
  logical,intent(in)::y(:,:)
  logical,intent(in)::p(:,:)
  class(*),optional,intent(in)::labels(:)
  real(dp),intent(in)::sample_weight(:)
  character(*),optional,intent(in)::normalize
  real(dp),allocatable::cm(:,:)
  cm=confusion_values(target_l(y),target_l(p),labels,sample_weight,normalize)
 end function
 function normalized_iv_iv(y,p,labels,normalize) result(cm)
  integer,intent(in)::y(:)
  integer,intent(in)::p(:)
  class(*),optional,intent(in)::labels(:)
  character(*),intent(in)::normalize
  real(dp),allocatable::cm(:,:)
  cm=confusion_values(target_i(y),target_i(p),labels,normalize=normalize)
 end function
 function normalized_iv_im(y,p,labels,normalize) result(cm)
  integer,intent(in)::y(:)
  integer,intent(in)::p(:,:)
  class(*),optional,intent(in)::labels(:)
  character(*),intent(in)::normalize
  real(dp),allocatable::cm(:,:)
  cm=confusion_values(target_i(y),target_i(p),labels,normalize=normalize)
 end function
 function normalized_iv_rv(y,p,labels,normalize) result(cm)
  integer,intent(in)::y(:)
  real(dp),intent(in)::p(:)
  class(*),optional,intent(in)::labels(:)
  character(*),intent(in)::normalize
  real(dp),allocatable::cm(:,:)
  cm=confusion_values(target_i(y),target_r(p),labels,normalize=normalize)
 end function
 function normalized_iv_rm(y,p,labels,normalize) result(cm)
  integer,intent(in)::y(:)
  real(dp),intent(in)::p(:,:)
  class(*),optional,intent(in)::labels(:)
  character(*),intent(in)::normalize
  real(dp),allocatable::cm(:,:)
  cm=confusion_values(target_i(y),target_r(p),labels,normalize=normalize)
 end function
 function normalized_iv_lv(y,p,labels,normalize) result(cm)
  integer,intent(in)::y(:)
  logical,intent(in)::p(:)
  class(*),optional,intent(in)::labels(:)
  character(*),intent(in)::normalize
  real(dp),allocatable::cm(:,:)
  cm=confusion_values(target_i(y),target_l(p),labels,normalize=normalize)
 end function
 function normalized_iv_lm(y,p,labels,normalize) result(cm)
  integer,intent(in)::y(:)
  logical,intent(in)::p(:,:)
  class(*),optional,intent(in)::labels(:)
  character(*),intent(in)::normalize
  real(dp),allocatable::cm(:,:)
  cm=confusion_values(target_i(y),target_l(p),labels,normalize=normalize)
 end function
 function normalized_im_iv(y,p,labels,normalize) result(cm)
  integer,intent(in)::y(:,:)
  integer,intent(in)::p(:)
  class(*),optional,intent(in)::labels(:)
  character(*),intent(in)::normalize
  real(dp),allocatable::cm(:,:)
  cm=confusion_values(target_i(y),target_i(p),labels,normalize=normalize)
 end function
 function normalized_im_im(y,p,labels,normalize) result(cm)
  integer,intent(in)::y(:,:)
  integer,intent(in)::p(:,:)
  class(*),optional,intent(in)::labels(:)
  character(*),intent(in)::normalize
  real(dp),allocatable::cm(:,:)
  cm=confusion_values(target_i(y),target_i(p),labels,normalize=normalize)
 end function
 function normalized_im_rv(y,p,labels,normalize) result(cm)
  integer,intent(in)::y(:,:)
  real(dp),intent(in)::p(:)
  class(*),optional,intent(in)::labels(:)
  character(*),intent(in)::normalize
  real(dp),allocatable::cm(:,:)
  cm=confusion_values(target_i(y),target_r(p),labels,normalize=normalize)
 end function
 function normalized_im_rm(y,p,labels,normalize) result(cm)
  integer,intent(in)::y(:,:)
  real(dp),intent(in)::p(:,:)
  class(*),optional,intent(in)::labels(:)
  character(*),intent(in)::normalize
  real(dp),allocatable::cm(:,:)
  cm=confusion_values(target_i(y),target_r(p),labels,normalize=normalize)
 end function
 function normalized_im_lv(y,p,labels,normalize) result(cm)
  integer,intent(in)::y(:,:)
  logical,intent(in)::p(:)
  class(*),optional,intent(in)::labels(:)
  character(*),intent(in)::normalize
  real(dp),allocatable::cm(:,:)
  cm=confusion_values(target_i(y),target_l(p),labels,normalize=normalize)
 end function
 function normalized_im_lm(y,p,labels,normalize) result(cm)
  integer,intent(in)::y(:,:)
  logical,intent(in)::p(:,:)
  class(*),optional,intent(in)::labels(:)
  character(*),intent(in)::normalize
  real(dp),allocatable::cm(:,:)
  cm=confusion_values(target_i(y),target_l(p),labels,normalize=normalize)
 end function
 function normalized_rv_iv(y,p,labels,normalize) result(cm)
  real(dp),intent(in)::y(:)
  integer,intent(in)::p(:)
  class(*),optional,intent(in)::labels(:)
  character(*),intent(in)::normalize
  real(dp),allocatable::cm(:,:)
  cm=confusion_values(target_r(y),target_i(p),labels,normalize=normalize)
 end function
 function normalized_rv_im(y,p,labels,normalize) result(cm)
  real(dp),intent(in)::y(:)
  integer,intent(in)::p(:,:)
  class(*),optional,intent(in)::labels(:)
  character(*),intent(in)::normalize
  real(dp),allocatable::cm(:,:)
  cm=confusion_values(target_r(y),target_i(p),labels,normalize=normalize)
 end function
 function normalized_rv_rv(y,p,labels,normalize) result(cm)
  real(dp),intent(in)::y(:)
  real(dp),intent(in)::p(:)
  class(*),optional,intent(in)::labels(:)
  character(*),intent(in)::normalize
  real(dp),allocatable::cm(:,:)
  cm=confusion_values(target_r(y),target_r(p),labels,normalize=normalize)
 end function
 function normalized_rv_rm(y,p,labels,normalize) result(cm)
  real(dp),intent(in)::y(:)
  real(dp),intent(in)::p(:,:)
  class(*),optional,intent(in)::labels(:)
  character(*),intent(in)::normalize
  real(dp),allocatable::cm(:,:)
  cm=confusion_values(target_r(y),target_r(p),labels,normalize=normalize)
 end function
 function normalized_rv_lv(y,p,labels,normalize) result(cm)
  real(dp),intent(in)::y(:)
  logical,intent(in)::p(:)
  class(*),optional,intent(in)::labels(:)
  character(*),intent(in)::normalize
  real(dp),allocatable::cm(:,:)
  cm=confusion_values(target_r(y),target_l(p),labels,normalize=normalize)
 end function
 function normalized_rv_lm(y,p,labels,normalize) result(cm)
  real(dp),intent(in)::y(:)
  logical,intent(in)::p(:,:)
  class(*),optional,intent(in)::labels(:)
  character(*),intent(in)::normalize
  real(dp),allocatable::cm(:,:)
  cm=confusion_values(target_r(y),target_l(p),labels,normalize=normalize)
 end function
 function normalized_rm_iv(y,p,labels,normalize) result(cm)
  real(dp),intent(in)::y(:,:)
  integer,intent(in)::p(:)
  class(*),optional,intent(in)::labels(:)
  character(*),intent(in)::normalize
  real(dp),allocatable::cm(:,:)
  cm=confusion_values(target_r(y),target_i(p),labels,normalize=normalize)
 end function
 function normalized_rm_im(y,p,labels,normalize) result(cm)
  real(dp),intent(in)::y(:,:)
  integer,intent(in)::p(:,:)
  class(*),optional,intent(in)::labels(:)
  character(*),intent(in)::normalize
  real(dp),allocatable::cm(:,:)
  cm=confusion_values(target_r(y),target_i(p),labels,normalize=normalize)
 end function
 function normalized_rm_rv(y,p,labels,normalize) result(cm)
  real(dp),intent(in)::y(:,:)
  real(dp),intent(in)::p(:)
  class(*),optional,intent(in)::labels(:)
  character(*),intent(in)::normalize
  real(dp),allocatable::cm(:,:)
  cm=confusion_values(target_r(y),target_r(p),labels,normalize=normalize)
 end function
 function normalized_rm_rm(y,p,labels,normalize) result(cm)
  real(dp),intent(in)::y(:,:)
  real(dp),intent(in)::p(:,:)
  class(*),optional,intent(in)::labels(:)
  character(*),intent(in)::normalize
  real(dp),allocatable::cm(:,:)
  cm=confusion_values(target_r(y),target_r(p),labels,normalize=normalize)
 end function
 function normalized_rm_lv(y,p,labels,normalize) result(cm)
  real(dp),intent(in)::y(:,:)
  logical,intent(in)::p(:)
  class(*),optional,intent(in)::labels(:)
  character(*),intent(in)::normalize
  real(dp),allocatable::cm(:,:)
  cm=confusion_values(target_r(y),target_l(p),labels,normalize=normalize)
 end function
 function normalized_rm_lm(y,p,labels,normalize) result(cm)
  real(dp),intent(in)::y(:,:)
  logical,intent(in)::p(:,:)
  class(*),optional,intent(in)::labels(:)
  character(*),intent(in)::normalize
  real(dp),allocatable::cm(:,:)
  cm=confusion_values(target_r(y),target_l(p),labels,normalize=normalize)
 end function
 function normalized_lv_iv(y,p,labels,normalize) result(cm)
  logical,intent(in)::y(:)
  integer,intent(in)::p(:)
  class(*),optional,intent(in)::labels(:)
  character(*),intent(in)::normalize
  real(dp),allocatable::cm(:,:)
  cm=confusion_values(target_l(y),target_i(p),labels,normalize=normalize)
 end function
 function normalized_lv_im(y,p,labels,normalize) result(cm)
  logical,intent(in)::y(:)
  integer,intent(in)::p(:,:)
  class(*),optional,intent(in)::labels(:)
  character(*),intent(in)::normalize
  real(dp),allocatable::cm(:,:)
  cm=confusion_values(target_l(y),target_i(p),labels,normalize=normalize)
 end function
 function normalized_lv_rv(y,p,labels,normalize) result(cm)
  logical,intent(in)::y(:)
  real(dp),intent(in)::p(:)
  class(*),optional,intent(in)::labels(:)
  character(*),intent(in)::normalize
  real(dp),allocatable::cm(:,:)
  cm=confusion_values(target_l(y),target_r(p),labels,normalize=normalize)
 end function
 function normalized_lv_rm(y,p,labels,normalize) result(cm)
  logical,intent(in)::y(:)
  real(dp),intent(in)::p(:,:)
  class(*),optional,intent(in)::labels(:)
  character(*),intent(in)::normalize
  real(dp),allocatable::cm(:,:)
  cm=confusion_values(target_l(y),target_r(p),labels,normalize=normalize)
 end function
 function normalized_lv_lv(y,p,labels,normalize) result(cm)
  logical,intent(in)::y(:)
  logical,intent(in)::p(:)
  class(*),optional,intent(in)::labels(:)
  character(*),intent(in)::normalize
  real(dp),allocatable::cm(:,:)
  cm=confusion_values(target_l(y),target_l(p),labels,normalize=normalize)
 end function
 function normalized_lv_lm(y,p,labels,normalize) result(cm)
  logical,intent(in)::y(:)
  logical,intent(in)::p(:,:)
  class(*),optional,intent(in)::labels(:)
  character(*),intent(in)::normalize
  real(dp),allocatable::cm(:,:)
  cm=confusion_values(target_l(y),target_l(p),labels,normalize=normalize)
 end function
 function normalized_lm_iv(y,p,labels,normalize) result(cm)
  logical,intent(in)::y(:,:)
  integer,intent(in)::p(:)
  class(*),optional,intent(in)::labels(:)
  character(*),intent(in)::normalize
  real(dp),allocatable::cm(:,:)
  cm=confusion_values(target_l(y),target_i(p),labels,normalize=normalize)
 end function
 function normalized_lm_im(y,p,labels,normalize) result(cm)
  logical,intent(in)::y(:,:)
  integer,intent(in)::p(:,:)
  class(*),optional,intent(in)::labels(:)
  character(*),intent(in)::normalize
  real(dp),allocatable::cm(:,:)
  cm=confusion_values(target_l(y),target_i(p),labels,normalize=normalize)
 end function
 function normalized_lm_rv(y,p,labels,normalize) result(cm)
  logical,intent(in)::y(:,:)
  real(dp),intent(in)::p(:)
  class(*),optional,intent(in)::labels(:)
  character(*),intent(in)::normalize
  real(dp),allocatable::cm(:,:)
  cm=confusion_values(target_l(y),target_r(p),labels,normalize=normalize)
 end function
 function normalized_lm_rm(y,p,labels,normalize) result(cm)
  logical,intent(in)::y(:,:)
  real(dp),intent(in)::p(:,:)
  class(*),optional,intent(in)::labels(:)
  character(*),intent(in)::normalize
  real(dp),allocatable::cm(:,:)
  cm=confusion_values(target_l(y),target_r(p),labels,normalize=normalize)
 end function
 function normalized_lm_lv(y,p,labels,normalize) result(cm)
  logical,intent(in)::y(:,:)
  logical,intent(in)::p(:)
  class(*),optional,intent(in)::labels(:)
  character(*),intent(in)::normalize
  real(dp),allocatable::cm(:,:)
  cm=confusion_values(target_l(y),target_l(p),labels,normalize=normalize)
 end function
 function normalized_lm_lm(y,p,labels,normalize) result(cm)
  logical,intent(in)::y(:,:)
  logical,intent(in)::p(:,:)
  class(*),optional,intent(in)::labels(:)
  character(*),intent(in)::normalize
  real(dp),allocatable::cm(:,:)
  cm=confusion_values(target_l(y),target_l(p),labels,normalize=normalize)
 end function
end module
