program reference_classification_datasets_driver
  use sklearn_datasets
  use sklearn_common, only: dp
  use, intrinsic :: ieee_arithmetic, only: ieee_value,ieee_quiet_nan,ieee_positive_inf
  implicit none
  type(classification_dataset_options) :: options
  real(dp), allocatable :: x(:,:)
  integer, allocatable :: y(:)
  integer :: family,n,p,h,s,rs,rc,nw,fs,fc,i
  real(dp) :: bad
  character(80) :: argument
  if(command_argument_count()>0) then
    call get_command_argument(1,argument);call reject(trim(argument));stop 2
  end if
  read(*,*) family,n,p,options%n_informative,options%n_redundant,options%n_repeated, &
    options%n_classes,options%n_clusters_per_class,options%random_state,h,s,rs,rc
  read(*,*) options%flip_y,options%class_sep,options%shift,options%scale
  options%hypercube=h/=0;options%shuffle=s/=0;options%random_shift=rs/=0;options%random_scale=rc/=0
  read(*,*) nw,fs,fc
  if(nw>=0) then
    allocate(options%weights(nw))
    if(nw>0) read(*,*) options%weights
  end if
  if(fs/=0) then
    allocate(options%feature_shift(p));read(*,*) options%feature_shift
  end if
  if(fc/=0) then
    allocate(options%feature_scale(p));read(*,*) options%feature_scale
  end if
  if(family==0) then
    call make_classification(n,p,x,y,options)
  else
    call make_hastie_10_2(n,x,y,options%random_state)
  end if
  print *,size(x,1),size(x,2)
  do i=1,size(x,2)
    print '(*(es26.17e3,1x))',x(:,i)
  end do
  print '(*(i0,1x))',y
contains
  subroutine reject(name)
    character(*), intent(in) :: name
    options=classification_dataset_options();n=12;p=6;bad=ieee_value(0._dp,ieee_quiet_nan)
    select case(name)
    case('samples');n=0
    case('features');p=0
    case('informative');options%n_informative=0
    case('redundant');options%n_redundant=-1
    case('repeated');options%n_repeated=-1
    case('blocks');options%n_repeated=3
    case('classes');options%n_classes=0
    case('clusters');options%n_clusters_per_class=0
    case('cluster_overflow');options%n_classes=huge(1);options%n_clusters_per_class=2
    case('vertices');options%n_classes=3
    case('negative_flip');options%flip_y=-.1_dp
    case('large_flip');options%flip_y=1.1_dp
    case('nan_flip');options%flip_y=bad
    case('zero_separation');options%class_sep=0
    case('nan_separation');options%class_sep=bad
    case('seed');options%random_state=-1
    case('weight_length');options%weights=[.1_dp,.2_dp,.3_dp]
    case('negative_weight');options%weights=[-.1_dp,.5_dp]
    case('negative_inferred');options%weights=[1.1_dp]
    case('nan_weight');options%weights=[bad,1._dp]
    case('large_weight');options%weights=[real(huge(1),dp),1._dp]
    case('shift_length');options%feature_shift=[1._dp]
    case('scale_length');options%feature_scale=[1._dp]
    case('nan_shift');options%shift=bad
    case('nan_scale');options%scale=bad
    case('zero_scale');options%scale=0
    case('negative_scale');options%scale=-1
    case('vector_shift_nan');options%feature_shift=[0._dp,0._dp,0._dp,0._dp,0._dp,bad]
    case('vector_scale_infinite')
      options%feature_scale=[1._dp,1._dp,1._dp,1._dp,1._dp,ieee_value(0._dp,ieee_positive_inf)]
    case('random_shift_conflict');options%random_shift=.true.;options%feature_shift=[1._dp]
    case('random_scale_conflict');options%random_scale=.true.;options%feature_scale=[1._dp]
    case('hastie_samples');call make_hastie_10_2(0,x,y);return
    case('hastie_seed');call make_hastie_10_2(3,x,y,-1);return
    case default;stop 2
    end select
    call make_classification(n,p,x,y,options)
  end subroutine
end program
