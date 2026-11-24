program test_multilabel_datasets
  use sklearn_common, only: dp,require
  use sklearn_datasets, only: make_multilabel_classification
  use sklearn_sparse, only: csr_matrix
  use sklearn_label_binarization, only: label_set
  implicit none
  real(dp), allocatable :: x(:,:),pc(:),pwc(:,:),other(:,:)
  integer, allocatable :: y(:,:),before(:),after(:)
  type(csr_matrix) :: sx,sy
  type(label_set), allocatable :: sets(:)
  integer :: i,j,nseed
  call make_multilabel_classification(31,13,x,y,random_state=42,p_c=pc,p_w_c=pwc)
  call require(all(shape(x)==[31,13]).and.all(shape(y)==[31,5]),'default multilabel shapes')
  call require(all(x>=0).and.all(x==aint(x)).and.all(sum(x,dim=2)>0),'nonempty integer word counts')
  call require(all(y==0.or.y==1),'binary indicator values')
  call require(abs(sum(pc)-1)<1.e-14_dp.and.maxval(abs(sum(pwc,dim=1)-1))<1.e-14_dp, &
    'class and conditional word probabilities normalize')
  call make_multilabel_classification(31,13,sx,sy,random_state=42)
  call require(all(sx%to_dense()==x).and.all(sy%to_dense()==real(y,dp)),'dense and CSR generation equality')
  call make_multilabel_classification(31,13,other,sets,random_state=42)
  call require(all(other==x),'label-set representation preserves feature draws')
  do i=1,31
    call require(size(sets(i)%values)==sum(y(i,:)),'label-set size equals indicator population')
    do j=1,size(sets(i)%values)
      call require(y(i,sets(i)%values(j)+1)==1,'label-set values use zero-based class labels')
    end do
  end do
  call make_multilabel_classification(17,7,x,y,n_classes=3,n_labels=0,length=1,random_state=7)
  call require(all(y==0).and.all(sum(x,dim=2)>0),'unlabeled noise documents retain positive length')
  call make_multilabel_classification(17,1,sx,sy,n_classes=1,n_labels=2,length=10, &
    allow_unlabeled=.false.,random_state=7)
  call require(all(sy%to_dense()==1).and.all(sx%to_dense()>0),'one feature/class with nonempty-label rejection')
  call random_seed(size=nseed);allocate(before(nseed),after(nseed));call random_seed(get=before)
  call make_multilabel_classification(3,100000,sx,sets,n_classes=2,length=3,random_state=0)
  call random_seed(get=after);call require(all(before==after),'multilabel generation leaves intrinsic RNG unchanged')
  call require(sx%nnz()<100.and.sx%n_cols==100000,'wide CSR generation retains only word-count nonzeros')
  print *, 'Multilabel Poisson sampling, dense/CSR/sets and isolated distributions: PASS'
end program
