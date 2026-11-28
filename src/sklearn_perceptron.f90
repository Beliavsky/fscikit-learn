! sklearn 1.6.1 linear_model/_perceptron.py (BSD-3-Clause).
! Perceptron fixes the shared SGD engine to perceptron loss/constant updates.
! Typed native storage/adapters are MIT; see THIRD_PARTY_NOTICES.md.
module sklearn_perceptron
  use sklearn_common,only:dp,require,check_x,unique_labels
  use sklearn_sgd,only:sgd_classifier
  use sklearn_sparse,only:csr_matrix
  use sklearn_metrics,only:accuracy_score
  use,intrinsic::ieee_arithmetic,only:ieee_is_finite
  implicit none
  private
  public::perceptron,clone_perceptron_parameters
  type::perceptron
    character(16)::penalty='none'
    real(dp)::alpha=1.e-4_dp,l1_ratio=.15_dp,eta0=1,tol=1.e-3_dp,validation_fraction=.1_dp
    integer::max_iter=1000,n_iter_no_change=5,random_state=0,n_iter_=0,n_features_in_=0
    logical::fit_intercept=.true.,shuffle=.true.,warm_start=.false.,early_stopping=.false.
    logical::class_weight_balanced=.false.,converged_=.false.
    real(dp)::t_=1
    integer,allocatable::classes_(:),n_iter_per_class_(:)
    logical,allocatable::validation_mask_(:),converged_per_class_(:)
    real(dp),allocatable::coef_(:,:),intercept_(:),loss_(:)
    type(sgd_classifier),private::engine_
  contains
    procedure,private::fit_dense_matrix
    procedure,private::fit_dense_vector
    procedure,private::fit_dense_scalar
    procedure,private::fit_dense_vector_scalar
    procedure,private::fit_csr_matrix
    procedure,private::fit_csr_vector
    procedure,private::fit_csr_scalar
    procedure,private::fit_csr_vector_scalar
    generic::fit=>fit_dense_matrix, &
      fit_dense_vector, &
      fit_dense_scalar, &
      fit_dense_vector_scalar, &
      fit_csr_matrix, &
      fit_csr_vector, &
      fit_csr_scalar, &
      fit_csr_vector_scalar
    procedure,private::partial_fit_dense,partial_fit_csr
    generic::partial_fit=>partial_fit_dense,partial_fit_csr
    procedure,private::decision_function_dense,decision_function_csr
    generic::decision_function=>decision_function_dense,decision_function_csr
    procedure,private::predict_dense,predict_csr
    generic::predict=>predict_dense,predict_csr
    procedure,private::score_dense,score_csr
    generic::score=>score_dense,score_csr
  end type
contains
  function clone_perceptron_parameters(source)result(model)
    type(perceptron),intent(in)::source
    type(perceptron)::model
    model%penalty=source%penalty;model%alpha=source%alpha;model%l1_ratio=source%l1_ratio
    model%eta0=source%eta0;model%tol=source%tol;model%validation_fraction=source%validation_fraction
    model%max_iter=source%max_iter;model%n_iter_no_change=source%n_iter_no_change;model%random_state=source%random_state
    model%fit_intercept=source%fit_intercept;model%shuffle=source%shuffle;model%warm_start=source%warm_start
    model%early_stopping=source%early_stopping;model%class_weight_balanced=source%class_weight_balanced
  end function
  subroutine configure(self)
    class(perceptron),intent(inout)::self
    self%engine_%loss='perceptron';self%engine_%learning_rate='constant';self%engine_%average=0
    self%engine_%penalty=self%penalty;self%engine_%alpha=self%alpha;self%engine_%l1_ratio=self%l1_ratio
    self%engine_%eta0=self%eta0;self%engine_%tol=self%tol;self%engine_%validation_fraction=self%validation_fraction
    self%engine_%max_iter=self%max_iter;self%engine_%n_iter_no_change=self%n_iter_no_change
    self%engine_%random_state=self%random_state;self%engine_%fit_intercept=self%fit_intercept
    self%engine_%shuffle=self%shuffle;self%engine_%warm_start=self%warm_start
    self%engine_%early_stopping=self%early_stopping;self%engine_%class_weight_balanced=self%class_weight_balanced
  end subroutine
  subroutine publish(self)
    class(perceptron),intent(inout)::self
    self%classes_=self%engine_%classes_;self%coef_=self%engine_%coef_;self%intercept_=self%engine_%intercept_
    self%n_iter_=self%engine_%n_iter_;self%n_features_in_=self%engine_%n_features_in_;self%t_=self%engine_%t_
    self%n_iter_per_class_=self%engine_%n_iter_per_class_;self%converged_=self%engine_%converged_
    self%converged_per_class_=self%engine_%converged_per_class_;self%loss_=self%engine_%loss_
    self%validation_mask_=self%engine_%validation_mask_
  end subroutine
  subroutine fit_dense_matrix(self,x,y,sample_weight,class_weight,coef_init,intercept_init)
    class(perceptron),intent(inout)::self
    real(dp),intent(in)::x(:,:)
    integer,intent(in)::y(:)
    real(dp),optional,intent(in)::sample_weight(:),class_weight(:)
    real(dp),optional,intent(in)::coef_init(:,:)
    real(dp),optional,intent(in)::intercept_init(:)
    call configure(self)
    if(self%warm_start.and.allocated(self%coef_))self%engine_%coef_=self%coef_
    if(self%warm_start.and.allocated(self%intercept_))self%engine_%intercept_=self%intercept_
    call self%engine_%fit(x,y,sample_weight,class_weight,coef_init,intercept_init)
    call publish(self)
  end subroutine
  subroutine fit_dense_vector(self,x,y,sample_weight,class_weight,coef_init,intercept_init)
    class(perceptron),intent(inout)::self
    real(dp),intent(in)::x(:,:)
    integer,intent(in)::y(:)
    real(dp),optional,intent(in)::sample_weight(:),class_weight(:)
    real(dp),intent(in)::coef_init(:)
    real(dp),optional,intent(in)::intercept_init(:)
    call configure(self)
    if(self%warm_start.and.allocated(self%coef_))self%engine_%coef_=self%coef_
    if(self%warm_start.and.allocated(self%intercept_))self%engine_%intercept_=self%intercept_
    call self%engine_%fit(x,y,sample_weight,class_weight,coef_init,intercept_init)
    call publish(self)
  end subroutine
  subroutine fit_dense_scalar(self,x,y,sample_weight,class_weight,coef_init,intercept_init)
    class(perceptron),intent(inout)::self
    real(dp),intent(in)::x(:,:)
    integer,intent(in)::y(:)
    real(dp),optional,intent(in)::sample_weight(:),class_weight(:)
    real(dp),optional,intent(in)::coef_init(:,:)
    real(dp),intent(in)::intercept_init
    call configure(self)
    if(self%warm_start.and.allocated(self%coef_))self%engine_%coef_=self%coef_
    if(self%warm_start.and.allocated(self%intercept_))self%engine_%intercept_=self%intercept_
    call self%engine_%fit(x,y,sample_weight,class_weight,coef_init,intercept_init)
    call publish(self)
  end subroutine
  subroutine fit_dense_vector_scalar(self,x,y,sample_weight,class_weight,coef_init,intercept_init)
    class(perceptron),intent(inout)::self
    real(dp),intent(in)::x(:,:)
    integer,intent(in)::y(:)
    real(dp),optional,intent(in)::sample_weight(:),class_weight(:)
    real(dp),intent(in)::coef_init(:)
    real(dp),intent(in)::intercept_init
    call configure(self)
    if(self%warm_start.and.allocated(self%coef_))self%engine_%coef_=self%coef_
    if(self%warm_start.and.allocated(self%intercept_))self%engine_%intercept_=self%intercept_
    call self%engine_%fit(x,y,sample_weight,class_weight,coef_init,intercept_init)
    call publish(self)
  end subroutine
  subroutine partial_fit_dense(self,x,y,classes,sample_weight,class_weight)
    class(perceptron),intent(inout)::self
    real(dp),intent(in)::x(:,:)
    integer,intent(in)::y(:)
    integer,optional,intent(in)::classes(:)
    real(dp),optional,intent(in)::sample_weight(:),class_weight(:)
    call configure(self)
    if(present(classes))then
      call self%engine_%partial_fit(x,y,unique_labels(classes),sample_weight,class_weight)
    else
      call self%engine_%partial_fit(x,y,sample_weight=sample_weight,class_weight=class_weight)
    end if
    call publish(self)
  end subroutine
  function decision_function_dense(self,x)result(z)
    class(perceptron),intent(in)::self
    real(dp),intent(in)::x(:,:)
    real(dp),allocatable::z(:,:)
    call require(allocated(self%coef_),'Perceptron is not fitted')
    call check_x(x)
    call require(size(x,2)==self%n_features_in_,'Perceptron feature count mismatch')
    z=matmul(x,transpose(self%coef_))+spread(self%intercept_,1,size(x,1))
  end function
  function predict_dense(self,x)result(z)
    class(perceptron),intent(in)::self
    real(dp),intent(in)::x(:,:)
    integer,allocatable::z(:)
    real(dp),allocatable::scores(:,:)
    scores=self%decision_function(x)
    if(size(self%classes_)==2)then
      z=self%classes_(merge(2,1,scores(:,1)>0))
    else
      z=self%classes_(maxloc(scores,dim=2))
    end if
  end function
  function score_dense(self,x,y,sample_weight)result(z)
    class(perceptron),intent(in)::self
    real(dp),intent(in)::x(:,:)
    real(dp)::z
    integer,intent(in)::y(:)
    real(dp),optional,intent(in)::sample_weight(:)
    z=accuracy_score(y,self%predict(x),sample_weight=sample_weight)
  end function
  subroutine fit_csr_matrix(self,x,y,sample_weight,class_weight,coef_init,intercept_init)
    class(perceptron),intent(inout)::self
    type(csr_matrix),intent(in)::x
    integer,intent(in)::y(:)
    real(dp),optional,intent(in)::sample_weight(:),class_weight(:)
    real(dp),optional,intent(in)::coef_init(:,:)
    real(dp),optional,intent(in)::intercept_init(:)
    call configure(self)
    if(self%warm_start.and.allocated(self%coef_))self%engine_%coef_=self%coef_
    if(self%warm_start.and.allocated(self%intercept_))self%engine_%intercept_=self%intercept_
    call self%engine_%fit(x,y,sample_weight,class_weight,coef_init,intercept_init)
    call publish(self)
  end subroutine
  subroutine fit_csr_vector(self,x,y,sample_weight,class_weight,coef_init,intercept_init)
    class(perceptron),intent(inout)::self
    type(csr_matrix),intent(in)::x
    integer,intent(in)::y(:)
    real(dp),optional,intent(in)::sample_weight(:),class_weight(:)
    real(dp),intent(in)::coef_init(:)
    real(dp),optional,intent(in)::intercept_init(:)
    call configure(self)
    if(self%warm_start.and.allocated(self%coef_))self%engine_%coef_=self%coef_
    if(self%warm_start.and.allocated(self%intercept_))self%engine_%intercept_=self%intercept_
    call self%engine_%fit(x,y,sample_weight,class_weight,coef_init,intercept_init)
    call publish(self)
  end subroutine
  subroutine fit_csr_scalar(self,x,y,sample_weight,class_weight,coef_init,intercept_init)
    class(perceptron),intent(inout)::self
    type(csr_matrix),intent(in)::x
    integer,intent(in)::y(:)
    real(dp),optional,intent(in)::sample_weight(:),class_weight(:)
    real(dp),optional,intent(in)::coef_init(:,:)
    real(dp),intent(in)::intercept_init
    call configure(self)
    if(self%warm_start.and.allocated(self%coef_))self%engine_%coef_=self%coef_
    if(self%warm_start.and.allocated(self%intercept_))self%engine_%intercept_=self%intercept_
    call self%engine_%fit(x,y,sample_weight,class_weight,coef_init,intercept_init)
    call publish(self)
  end subroutine
  subroutine fit_csr_vector_scalar(self,x,y,sample_weight,class_weight,coef_init,intercept_init)
    class(perceptron),intent(inout)::self
    type(csr_matrix),intent(in)::x
    integer,intent(in)::y(:)
    real(dp),optional,intent(in)::sample_weight(:),class_weight(:)
    real(dp),intent(in)::coef_init(:)
    real(dp),intent(in)::intercept_init
    call configure(self)
    if(self%warm_start.and.allocated(self%coef_))self%engine_%coef_=self%coef_
    if(self%warm_start.and.allocated(self%intercept_))self%engine_%intercept_=self%intercept_
    call self%engine_%fit(x,y,sample_weight,class_weight,coef_init,intercept_init)
    call publish(self)
  end subroutine
  subroutine partial_fit_csr(self,x,y,classes,sample_weight,class_weight)
    class(perceptron),intent(inout)::self
    type(csr_matrix),intent(in)::x
    integer,intent(in)::y(:)
    integer,optional,intent(in)::classes(:)
    real(dp),optional,intent(in)::sample_weight(:),class_weight(:)
    call configure(self)
    if(present(classes))then
      call self%engine_%partial_fit(x,y,unique_labels(classes),sample_weight,class_weight)
    else
      call self%engine_%partial_fit(x,y,sample_weight=sample_weight,class_weight=class_weight)
    end if
    call publish(self)
  end subroutine
  function decision_function_csr(self,x)result(z)
    class(perceptron),intent(in)::self
    type(csr_matrix),intent(in)::x
    real(dp),allocatable::z(:,:)
    call require(allocated(self%coef_),'Perceptron is not fitted')
    call x%validate()
    call require(x%n_rows>0.and.x%n_cols==self%n_features_in_,'Perceptron feature count mismatch')
    call require(all(ieee_is_finite(x%data)),'nonfinite Perceptron CSR data')
    z=x%matmul(transpose(self%coef_))+spread(self%intercept_,1,x%n_rows)
  end function
  function predict_csr(self,x)result(z)
    class(perceptron),intent(in)::self
    type(csr_matrix),intent(in)::x
    integer,allocatable::z(:)
    real(dp),allocatable::scores(:,:)
    scores=self%decision_function(x)
    if(size(self%classes_)==2)then
      z=self%classes_(merge(2,1,scores(:,1)>0))
    else
      z=self%classes_(maxloc(scores,dim=2))
    end if
  end function
  function score_csr(self,x,y,sample_weight)result(z)
    class(perceptron),intent(in)::self
    type(csr_matrix),intent(in)::x
    real(dp)::z
    integer,intent(in)::y(:)
    real(dp),optional,intent(in)::sample_weight(:)
    z=accuracy_score(y,self%predict(x),sample_weight=sample_weight)
  end function
end module
