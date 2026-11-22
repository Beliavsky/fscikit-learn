program reference_discrete_stream_driver
  use sklearn_common, only: dp
  use sklearn_naive_bayes, only: multinomial_nb,bernoulli_nb,complement_nb
  use, intrinsic :: ieee_arithmetic, only: ieee_value,ieee_quiet_nan,ieee_positive_inf
  implicit none
  class(multinomial_nb), allocatable :: model
  real(dp), allocatable :: x(:,:),q(:,:),weights(:),saved(:,:)
  integer, allocatable :: y(:),classes(:),ends(:)
  integer :: kind,n,p,m,k,stream,batches,weighted,fitprior,fixed,norm,i,start
  real(dp) :: alpha,threshold
  character(80) :: arg
  if(command_argument_count()>0) then
    call get_command_argument(1,arg);call reject(trim(arg));stop 2
  end if
  read(*,*) kind,n,p,m,k,stream,batches,weighted,fitprior,fixed,norm,alpha,threshold
  select case(kind)
  case(0);allocate(multinomial_nb::model)
  case(1);allocate(bernoulli_nb::model)
  case(2);allocate(complement_nb::model)
  end select
  allocate(x(n,p),q(m,p),weights(n),y(n),classes(k),ends(batches))
  read(*,*) x;read(*,*) y;read(*,*) q;read(*,*) weights;read(*,*) classes;read(*,*) ends
  model%alpha=alpha;model%fit_prior=fitprior==1
  if(fixed==1) then
    allocate(model%class_prior(k));read(*,*) model%class_prior
  end if
  select type(model)
  type is(bernoulli_nb);model%binarize_threshold=threshold
  type is(complement_nb);model%norm=norm==1
  end select
  saved=x
  if(stream==0) then
    if(weighted==1) then
      call model%fit(x,y,weights)
    else
      call model%fit(x,y)
    end if
    call snapshot()
  else
    start=1
    do i=1,batches
      if(i==1.or.mod(i,2)==0) then
        if(weighted==1) then
          call model%partial_fit(x(start:ends(i),:),y(start:ends(i)),classes,weights(start:ends(i)))
        else
          call model%partial_fit(x(start:ends(i),:),y(start:ends(i)),classes)
        end if
      else
        if(weighted==1) then
          call model%partial_fit(x(start:ends(i),:),y(start:ends(i)),sample_weight=weights(start:ends(i)))
        else
          call model%partial_fit(x(start:ends(i),:),y(start:ends(i)))
        end if
      end if
      call snapshot();start=ends(i)+1
    end do
  end if
  if(any(x/=saved)) stop 4
contains
  subroutine vector(a)
    real(dp), intent(in) :: a(:)
    print *,size(a);print '(*(es26.17e3,1x))',a
  end subroutine
  subroutine matrix(a)
    real(dp), intent(in) :: a(:,:)
    print *,shape(a);print '(*(es26.17e3,1x))',reshape(a,[size(a)])
  end subroutine
  subroutine snapshot()
    call vector(real(model%classes_,dp));call matrix(model%feature_count_);call vector(model%class_count_)
    call matrix(model%feature_log_prob_);call vector(model%class_log_prior_)
    call matrix(model%joint_log_likelihood(q));call matrix(model%predict_log_proba(q));call matrix(model%predict_proba(q))
    call vector(real(model%predict(q),dp));call vector([real(model%n_features_in_,dp)]);call vector([model%score(x,y)]);call vector(real(model%predict(x),dp))
    select type(model)
    type is(complement_nb);call vector(model%feature_all_)
    end select
  end subroutine
  subroutine reject(name)
    character(*), intent(in) :: name
    type(multinomial_nb) :: mn
    type(bernoulli_nb) :: bn
    real(dp) :: data(4,3),weight(4),bad
    real(dp), allocatable :: result(:,:)
    integer :: labels(4)
    data=1;weight=1;labels=[-2,-2,3,3];bad=ieee_value(0._dp,ieee_quiet_nan)
    select case(name)
    case('first_classes');call mn%partial_fit(data,labels);return
    case('empty_classes');call mn%partial_fit(data,labels,[integer::]);return
    case('changed_classes');call mn%partial_fit(data,labels,[-2,3]);call mn%partial_fit(data,labels,[-2,4]);return
    case('changed_width');call mn%partial_fit(data,labels,[-2,3]);call mn%partial_fit(data(:,:2),labels);return
    case('unknown_label');call mn%partial_fit(data,labels,[-2,4]);return
    case('alpha_zero');mn%alpha=0
    case('alpha_nan');mn%alpha=bad
    case('negative_x');data(1,1)=-1
    case('nan_x');data(1,1)=bad
    case('negative_weight');weight(1)=-1
    case('nan_weight');weight(1)=bad
    case('weight_size');call mn%fit(data,labels,weight(:3));return
    case('prior_size');mn%class_prior=[1._dp]
    case('prior_negative');mn%class_prior=[-1._dp,2._dp]
    case('prior_nan');mn%class_prior=[bad,1._dp]
    case('prior_inf');mn%class_prior=[ieee_value(0._dp,ieee_positive_inf),1._dp]
    case('prior_zero');mn%class_prior=[0._dp,0._dp]
    case('threshold_nan');bn%binarize_threshold=bad;call bn%fit(data,labels);return
    case('threshold_negative');bn%binarize_threshold=-.5_dp;call bn%fit(data,labels);return
    case('bernoulli_nan');data(1,1)=bad;call bn%fit(data,labels);return
    case('unfitted');result=mn%predict_proba(data);return
    case('query_width');call mn%fit(data,labels);result=mn%predict_proba(data(:,:2));return
    case('query_negative');call mn%fit(data,labels);data(1,1)=-1;result=mn%predict_proba(data);return
    case('query_nan');call mn%fit(data,labels);data(1,1)=bad;result=mn%predict_proba(data);return
    case('overflow_product');data=2;weight=huge(1._dp)
    case default;stop 3
    end select
    call mn%fit(data,labels,weight)
  end subroutine
end program
