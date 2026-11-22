program reference_model_selector_driver
  use sklearn_model_selection_features, only: select_from_model
  use sklearn_common, only: dp
  use sklearn_linear_model, only: linear_regression,ridge,lasso,logistic_regression, &
    multi_task_elastic_net,multi_task_lasso,sgd_regressor,sgd_classifier
  use sklearn_tree, only: decision_tree_regressor,decision_tree_classifier
  use sklearn_feature_extraction, only: feature_string
  implicit none
  type(select_from_model) :: selector
  type(linear_regression) :: linear
  type(ridge) :: regularized
  type(lasso) :: sparse
  type(logistic_regression) :: classifier
  type(multi_task_elastic_net) :: matrix_base
  type(multi_task_lasso) :: matrix_sparse
  type(sgd_regressor) :: stochastic_reg
  type(sgd_classifier) :: stochastic_clf
  type(decision_tree_regressor) :: tree_reg
  type(decision_tree_classifier) :: tree_clf
  type(feature_string), allocatable :: input(:),names(:),stream_names(:)
  real(dp), allocatable :: x(:,:),q(:,:),y(:),ym(:,:),w(:),coefficients(:,:),fit(:,:),query(:,:),inverse(:,:),scores(:)
  integer, allocatable :: yi(:),indices(:)
  integer :: mode,n,p,m,r,target_format,named,weighted,cap,cap_callback,operation,i,j,lo,hi
  integer :: importance_calls=0,cap_calls=0
  logical :: callback_matrix
  character(32) :: buffer
  read(*,*) mode,n,p,m,r,target_format,named,weighted,cap,cap_callback,operation
  read(*,'(a)') selector%threshold
  read(*,*) selector%numeric_threshold,selector%norm_order
  allocate(x(n,p),q(m,p),y(n),ym(n,r),yi(n),w(n),coefficients(r,p),input(p))
  read(*,*) x;read(*,*) q;read(*,*) y;read(*,*) yi;read(*,*) ym;read(*,*) w;read(*,*) coefficients
  do i=1,p
    write(buffer,'(a,i0)') 'field_',i;input(i)%text=trim(buffer)
  end do
  selector%max_features=cap
  if(cap_callback==1) selector%max_features_func=>feature_cap
  select case(mode)
  case(0)
    linear%coef_=coefficients(1,:);allocate(selector%estimator,source=linear);selector%prefit=.true.
  case(1)
    matrix_base%coef_=coefficients;allocate(selector%estimator,source=matrix_base);selector%prefit=.true.
  case(2)
    regularized%alpha=.7_dp;allocate(selector%estimator,source=regularized)
  case(3)
    sparse%alpha=.08_dp;sparse%tol=1.e-12_dp;allocate(selector%estimator,source=sparse)
  case(4)
    classifier%c=.8_dp;classifier%tol=1.e-11_dp;allocate(selector%estimator,source=classifier)
  case(5)
    tree_reg%max_depth=3;tree_reg%random_state=7;allocate(selector%estimator,source=tree_reg)
  case(6)
    tree_clf%max_depth=3;tree_clf%random_state=7;allocate(selector%estimator,source=tree_clf)
  case(7)
    matrix_sparse%alpha=.08_dp;matrix_sparse%tol=1.e-12_dp;allocate(selector%estimator,source=matrix_sparse)
  case(8)
    stochastic_reg%penalty='l2';stochastic_reg%alpha=.02_dp;stochastic_reg%learning_rate='constant'
    stochastic_reg%eta0=.01_dp;stochastic_reg%tol=-1;allocate(selector%estimator,source=stochastic_reg)
  case(9)
    stochastic_clf%loss='log_loss';stochastic_clf%penalty='l2';stochastic_clf%alpha=.02_dp
    stochastic_clf%learning_rate='constant';stochastic_clf%eta0=.01_dp;stochastic_clf%tol=-1
    allocate(selector%estimator,source=stochastic_clf)
  case(10,11)
    linear%coef_=coefficients(1,:);allocate(selector%estimator,source=linear);selector%prefit=.true.
    selector%importance_getter='callback';selector%importance_func=>importance
  end select
  if(selector%threshold=='auto'.and.cap==p) then
    if(mode<=1.or.mode==2.or.mode==3.or.mode==4.or.mode==7.or.mode==8.or.mode==9) selector%importance_getter='coef_'
    if(mode==5.or.mode==6) selector%importance_getter='feature_importances_'
  end if
  if(named==1) stream_names=input
  if(mode==8.or.mode==9) then
    do j=1,3
      lo=(j-1)*n/3+1;hi=j*n/3
      if(mode==8) then
        if(weighted==1) then
          call selector%partial_fit(x(lo:hi,:),y(lo:hi),w(lo:hi),feature_names=stream_names)
        else
          call selector%partial_fit(x(lo:hi,:),y(lo:hi),feature_names=stream_names)
        end if
      else
        if(weighted==1) then
          call selector%partial_fit(x(lo:hi,:),yi(lo:hi),[-4,9],w(lo:hi),feature_names=stream_names)
        else
          call selector%partial_fit(x(lo:hi,:),yi(lo:hi),[-4,9],feature_names=stream_names)
        end if
      end if
    end do
    fit=selector%transform(x)
  else if(operation==2) then
    fit=selector%transform(x)
  else
    call train()
    if(.not.allocated(fit)) fit=selector%transform(x)
  end if
  query=selector%transform(q)
  if(size(query,2)>0) then
    inverse=selector%inverse_transform(query)
  else
    allocate(inverse(0,0))
  end if
  indices=selector%get_support_indices();scores=selector%feature_importances()
  if(operation==2) then
    allocate(names(0))
  else
    names=selector%get_feature_names_out()
  end if
  write(*,*) size(indices),selector%n_features_in_,selector%max_features_
  write(*,'(*(es26.17e3,1x))') selector%threshold_value()
  write(*,*) indices
  write(*,'(*(es26.17e3,1x))') scores
  call emit(fit);call emit(query);call emit(inverse)
  write(*,*) size(names)
  do i=1,size(names)
    write(*,'(a)') names(i)%text
  end do
  write(*,*) importance_calls,cap_calls
  select type(model=>selector%estimator_)
  type is(decision_tree_regressor)
    write(*,*) model%node_count_
    do i=1,model%node_count_
      associate(node=>model%nodes_(i))
        write(*,*) node%left,node%right,node%feature,node%threshold,node%impurity,node%weighted_n_samples
      end associate
    end do
  type is(decision_tree_classifier)
    write(*,*) model%node_count_
    do i=1,model%node_count_
      associate(node=>model%nodes_(i))
        write(*,*) node%left,node%right,node%feature,node%threshold,node%impurity,node%weighted_n_samples
      end associate
    end do
  class default
    write(*,*) 0
  end select
contains
  subroutine train()
    type(feature_string), allocatable :: features(:)
    real(dp), allocatable :: weights(:)
    if(named==1) features=input
    if(weighted==1) weights=w
    if(mode==1.or.mode==7) then
      if(operation==1) then
        fit=selector%fit_transform(x,ym,feature_names=features)
      else
        call selector%fit(x,ym,feature_names=features)
      end if
    else if(mode==0.or.mode==10.or.mode==11) then
      if(operation==1) then
        fit=selector%fit_transform(x,feature_names=features)
      else
        call selector%fit(x,feature_names=features)
      end if
    else if(target_format==1) then
      if(operation==1) then
        fit=selector%fit_transform(x,yi,sample_weight=weights,feature_names=features)
      else
        call selector%fit(x,yi,sample_weight=weights,feature_names=features)
      end if
    else
      if(operation==1) then
        fit=selector%fit_transform(x,y,sample_weight=weights,feature_names=features)
      else
        call selector%fit(x,y,sample_weight=weights,feature_names=features)
      end if
    end if
  end subroutine
  integer function feature_cap(data) result(count)
    real(dp), intent(in) :: data(:,:)
    cap_calls=cap_calls+1;count=size(data,2)/2
  end function
  subroutine importance(model,values,matrix)
    class(*), intent(in) :: model
    real(dp), allocatable, intent(out) :: values(:,:)
    logical, intent(out) :: matrix
    importance_calls=importance_calls+1;values=coefficients;matrix=mode==11
  end subroutine
  subroutine emit(values)
    real(dp), intent(in) :: values(:,:)
    integer :: k
    write(*,*) shape(values)
    do k=1,size(values,1)
      write(*,'(*(es26.17e3,1x))') values(k,:)
    end do
  end subroutine
end program
