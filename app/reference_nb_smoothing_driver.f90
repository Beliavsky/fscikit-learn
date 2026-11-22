program reference_nb_smoothing_driver
  use sklearn_common, only: dp
  use sklearn_naive_bayes, only: multinomial_nb,bernoulli_nb,complement_nb
  use, intrinsic :: ieee_arithmetic, only: ieee_value,ieee_quiet_nan,ieee_positive_inf
  implicit none
  class(multinomial_nb), allocatable :: model
  real(dp), allocatable :: x(:,:),q(:,:),weights(:),saved(:,:)
  integer, allocatable :: y(:),classes(:),ends(:)
  integer :: kind,n,p,m,k,stream,batches,weighted,fitprior,fixed,norm,i,start,force,vector_alpha
  real(dp) :: alpha,threshold
  character(80) :: arg
  if(command_argument_count()>0) then
    call get_command_argument(1,arg);call reject(trim(arg));stop 2
  end if
  read(*,*) kind,n,p,m,k,stream,batches,weighted,fitprior,fixed,norm,alpha,threshold,force,vector_alpha
  select case(kind)
  case(0);allocate(multinomial_nb::model)
  case(1);allocate(bernoulli_nb::model)
  case(2);allocate(complement_nb::model)
  end select
  allocate(x(n,p),q(m,p),weights(n),y(n),classes(k),ends(batches))
  read(*,*) x;read(*,*) y;read(*,*) q;read(*,*) weights;read(*,*) classes;read(*,*) ends
  model%alpha=alpha;model%fit_prior=fitprior==1;model%force_alpha=force==1
  if(fixed==1) then
    allocate(model%class_prior(k));read(*,*) model%class_prior
  end if
  if(vector_alpha==1) then
    allocate(model%alpha_per_feature(p));read(*,*) model%alpha_per_feature
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
    type(complement_nb) :: cn
    real(dp) :: data(4,3),bad
    integer :: labels(4)
    data=1;labels=[-2,-2,3,3];bad=ieee_value(0._dp,ieee_quiet_nan)
    select case(name)
    case('vector_size');mn%alpha_per_feature=[1._dp,2._dp]
    case('vector_empty');mn%alpha_per_feature=[real(dp)::]
    case('vector_negative');mn%alpha_per_feature=[1._dp,-1._dp,1._dp]
    case('vector_nan');mn%alpha_per_feature=[1._dp,bad,1._dp]
    case('vector_inf');mn%alpha_per_feature=[1._dp,ieee_value(0._dp,ieee_positive_inf),1._dp]
    case('bernoulli_vector');bn%alpha_per_feature=[1._dp,1._dp,1._dp];call bn%fit(data,labels);return
    case('scalar_range');mn%alpha=huge(1._dp)
    case('bernoulli_range');bn%alpha=huge(1._dp);call bn%fit(data,labels);return
    case('complement_range');cn%alpha=huge(1._dp);call cn%fit(data,labels);return
    case('vector_range');mn%alpha_per_feature=[huge(1._dp),huge(1._dp),huge(1._dp)]
    case('complement_vector_range')
      cn%alpha_per_feature=[huge(1._dp),huge(1._dp),huge(1._dp)];call cn%fit(data,labels);return
    case default;error stop 'unknown smoothing rejection case'
    end select
    call mn%fit(data,labels)
  end subroutine
end program
