program test_bicluster
  use sklearn, only: dp,spectral_coclustering,spectral_biclustering
  use sklearn_common, only: require
  use sklearn_legacy_rng, only: legacy_random_state
  implicit none
  type(spectral_coclustering) :: co
  type(spectral_biclustering) :: bi
  type(legacy_random_state) :: rng
  real(dp) :: x(8,6),gram(3,3),identity(3,3)
  real(dp), allocatable :: values(:,:),tall(:,:)
  logical, allocatable :: rows(:,:),columns(:,:)
  integer, allocatable :: ri(:),ci(:),before(:)
  integer :: i,j,method,block,coverage(8,6)
  character(16) :: methods(3)=[character(16)::'scale','bistochastic','log']
  co=spectral_coclustering(); bi=spectral_biclustering()
  call rng%seed(42)
  do j=1,6
    do i=1,8
      if(i<=4) then
        x(i,j)=2
        if(j<=3) x(i,j)=12
      else
        x(i,j)=3
        if(j>3) x(i,j)=15
      end if
      x(i,j)=x(i,j)+.001_dp*rng%uniform()
    end do
  end do
  co%n_clusters=2; co%random_state=42; call co%fit(x)
  call require(all(co%row_labels_(:4)==co%row_labels_(1)),'coclustering first row block')
  call require(all(co%row_labels_(5:)==co%row_labels_(5)),'coclustering second row block')
  call require(co%row_labels_(1)/=co%row_labels_(5),'coclustering distinct row blocks')
  call require(all(co%column_labels_(:3)==co%row_labels_(1)),'coclustering paired first column block')
  call require(all(co%column_labels_(4:)==co%row_labels_(5)),'coclustering paired second column block')
  call co%get_indices(co%row_labels_(1),ri,ci)
  call require(all(ri==[1,2,3,4]).and.all(ci==[1,2,3]),'bicluster one-based extracted indices')
  call require(all(co%get_shape(co%row_labels_(1))==[4,3]),'bicluster extracted shape')
  values=co%get_submatrix(co%row_labels_(1),x)
  call require(all(values==x(ri,ci)),'bicluster extracted original data')
  call co%get_biclusters(rows,columns)
  call require(all(rows.eqv.co%rows_).and.all(columns.eqv.co%columns_),'bicluster paired indicators')
  before=co%row_labels_; call co%fit(x)
  call require(all(before==co%row_labels_),'coclustering repeated seeded fit')
  call co%fit(transpose(x))
  call require(co%n_rows_==6.and.co%n_features_in_==8,'coclustering wide SVD and refit shapes')
  do method=1,3
    bi%n_clusters=2; bi%n_components=1; bi%n_best=1; bi%method=methods(method); call bi%fit(x)
    call require(all(bi%row_labels_(:4)==bi%row_labels_(1)),'biclustering first row block')
    call require(all(bi%row_labels_(5:)==bi%row_labels_(5)),'biclustering second row block')
    call require(bi%row_labels_(1)/=bi%row_labels_(5),'biclustering distinct row blocks')
    call require(all(bi%column_labels_(:3)==bi%column_labels_(1)),'biclustering first column block')
    call require(all(bi%column_labels_(4:)==bi%column_labels_(4)),'biclustering second column block')
    call require(bi%column_labels_(1)/=bi%column_labels_(4),'biclustering distinct column blocks')
    coverage=0
    do block=1,4
      coverage=coverage+spread(merge(1,0,bi%rows_(block,:)),2,6)*spread(merge(1,0,bi%columns_(block,:)),1,8)
    end do
    call require(all(coverage==1),'bicluster Cartesian grid covers each cell once')
    if(method==3) then
      call require(maxval(abs(sum(bi%normalized_data_,dim=1)))<1.e-11_dp,'log bicluster centered columns')
      call require(maxval(abs(sum(bi%normalized_data_,dim=2)))<1.e-11_dp,'log bicluster centered rows')
    end if
  end do
  bi%n_components=3; bi%n_best=2; call bi%fit(0*x+2)
  identity=0
  do i=1,3
    identity(i,i)=1
  end do
  gram=matmul(transpose(bi%left_vectors_),bi%left_vectors_)
  call require(maxval(abs(gram-identity))<1.e-11_dp,'bicluster completed left null-space basis')
  gram=matmul(transpose(bi%right_vectors_),bi%right_vectors_)
  call require(maxval(abs(gram-identity))<1.e-11_dp,'bicluster orthogonal right null-space basis')
  call require(all(bi%row_labels_==0).and.all(bi%column_labels_==0),'bicluster duplicate projections')
  values=bi%get_submatrix(3,x)
  call require(all(shape(values)==[0,0]),'bicluster empty block submatrix')
  bi%n_components=20; bi%n_best=10; bi%method='scale'; call bi%fit(x)
  call require(size(bi%best_left_indices_)==5,'bicluster singular component truncation')
  bi%mini_batch=.true.; bi%n_components=2; bi%n_best=1; call bi%fit(x-20)
  call require(bi%n_biclusters_==4.and.all(shape(bi%normalized_data_)==[8,6]),'bicluster mini-batch negative input')
  allocate(tall(10000,4)); tall=2
  bi%mini_batch=.false.; bi%method='log'; bi%n_components=3; bi%n_best=1; call bi%fit(tall)
  gram=matmul(transpose(bi%left_vectors_),bi%left_vectors_)
  call require(maxval(abs(gram-identity))<1.e-11_dp,'bicluster tall null-space completion')
  call require(all(bi%row_labels_==0).and.all(bi%column_labels_==0),'bicluster tall constant projections')
  print *, 'Spectral co-/biclustering tests passed'
end program
