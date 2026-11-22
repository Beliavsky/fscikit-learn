program test_classification_datasets
  use sklearn, only: dp,make_classification,make_hastie_10_2,classification_dataset_options
  use sklearn_common, only: require
  use sklearn_legacy_rng, only: legacy_random_state
  use sklearn_bagging_sampling, only: sample_indices
  implicit none
  type(classification_dataset_options) :: options
  type(legacy_random_state) :: rng
  real(dp), allocatable :: x(:,:),a(:,:)
  integer, allocatable :: y(:),b(:),ids(:)
  integer :: j
  call make_classification(31,8,x,y)
  call make_classification(31,8,a,b,classification_dataset_options())
  call require(all(x==a).and.all(y==b),'classification deterministic defaults')
  options%n_redundant=0;options%n_repeated=4;options%shuffle=.false.;options%flip_y=0
  call make_classification(31,6,x,y,options)
  do j=3,6
    call require(all(x(:,j)==x(:,1)).or.all(x(:,j)==x(:,2)),'classification repeated columns')
  end do
  call require(all(y(:8)==0).and.all(y(9:16)==1).and.all(y(17:24)==0), &
    'classification assigns remainder samples to earliest clusters')
  options=classification_dataset_options();options%n_classes=3;options%n_informative=3
  options%weights=[.2_dp,.3_dp];options%random_shift=.true.;options%random_scale=.true.
  call make_classification(31,8,x,y,options)
  call require(size(options%weights)==2.and.options%weights(1)==.2_dp,'classification leaves input options unchanged')
  call require(all(y>=0).and.all(y<3),'classification labels are zero based')
  options%n_informative=40
  call make_classification(7,43,x,y,options)
  call require(all(shape(x)==[7,43]),'hypercube informative dimension above 30')
  call rng%seed(42);ids=sample_indices(rng,2**30,4,.false.)
  do j=1,4
    call require(count(ids==ids(j))==1,'large population tracking samples are distinct')
  end do
  call make_hastie_10_2(101,x,y,42)
  call require(all(y==merge(1,-1,sum(x**2,dim=2)>9.34_dp)),'Hastie radial class boundary')
  print *, 'Classification sample generators: PASS'
end program
